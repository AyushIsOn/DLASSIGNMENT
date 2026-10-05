"""Safety-first RAG orchestration with extractive-default generation gates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from acharya.config import Settings
from acharya.providers.base import (
    GenerativeProvider,
    ProviderContext,
    ProviderError,
    ProviderRequest,
)
from acharya.providers.extractive import ExtractiveProvider
from acharya.rag.embed import DenseEncoder, Reranker
from acharya.rag.evaluate import (
    is_ready,
    load_calibration,
    load_runtime_models,
    load_support_calibration,
    support_is_ready,
)
from acharya.rag.grounding import (
    ClaimScorer,
    GroundingError,
    TransformersClaimScorer,
    validate_extractive_result,
    validate_generated_result,
)
from acharya.rag.index import BM25Index, RetrievalMode, load_index
from acharya.rag.prompt import PromptBudget, correction_prompt, render_grounded_prompt
from acharya.rag.retrieve import RetrievalHit, retrieve
from acharya.safety import SafetyAction, SafetyPolicy
from acharya.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    GateState,
    GateStatus,
    HealthResponse,
    RetrievalScores,
)


class ServiceUnavailable(RuntimeError):
    """Raised when calibrated local retrieval is unavailable."""


class RAGService:
    def __init__(
        self,
        workspace: Path | str,
        *,
        generative_provider: GenerativeProvider | None = None,
        generation_mode: Literal["extractive", "optional"] = "extractive",
        support_scorer: ClaimScorer | None = None,
    ) -> None:
        self.settings = Settings.load(workspace)
        self.policy = SafetyPolicy(self.settings.safety)
        extractive = self.settings.rag["extractive"]
        self.provider = ExtractiveProvider(
            int(extractive["max_sentences"]), int(extractive["max_characters"])
        )
        self.generative_provider = generative_provider
        self.generation_mode = generation_mode
        self._support_scorer = support_scorer
        self.last_retrieval_hits: tuple[RetrievalHit, ...] = ()
        self.last_failure_reason: str | None = None
        self._runtime_fingerprint: str | None = None
        self._embedder: DenseEncoder | None = None
        self._reranker: Reranker | None = None

    @property
    def generative_ready(self) -> bool:
        return (
            self.generation_mode == "optional"
            and self.generative_provider is not None
            and support_is_ready(self.settings)
        )

    def gate_statuses(self) -> tuple[GateStatus, GateStatus, GateStatus]:
        ready = is_ready(self.settings)
        gate_b = GateStatus(gate="B", status=GateState.PENDING, reason="full rebuild not executed")
        gate_b_path = self.settings.workspace / "artifacts" / "gates" / "gate-b.json"
        if gate_b_path.is_file():
            try:
                evidence = json.loads(gate_b_path.read_text(encoding="utf-8"))
                if evidence.get("status") == "BLOCKED_EXTERNAL_DATA":
                    gate_b = GateStatus(
                        gate="B",
                        status=GateState.BLOCKED,
                        evidence=(str(evidence.get("component", "dataset_acquisition")),),
                        reason="external dataset acquisition is blocked",
                    )
                elif evidence.get("status") == "PASSED":
                    gate_b = GateStatus(
                        gate="B",
                        status=GateState.PASSED,
                        evidence=("strict_data", "full_retrieval", "transfer_bundle"),
                    )
                elif evidence.get("status") == "DATA_PREPARED":
                    gate_b = GateStatus(
                        gate="B",
                        status=GateState.PENDING,
                        evidence=("datasets_prepared",),
                        reason="remaining Gate B components are pending",
                    )
            except (OSError, ValueError, TypeError):
                pass
        gate_c = GateStatus(
            gate="C",
            status=GateState.PENDING_EXTERNAL_GPU,
            reason="optional external GPU training not executed",
        )
        gate_c_path = self.settings.workspace / "artifacts" / "gates" / "gate-c.json"
        if gate_c_path.is_file():
            try:
                gate_c_evidence = json.loads(gate_c_path.read_text(encoding="utf-8"))
                if gate_c_evidence.get("status") == "COMPLETED":
                    gate_c = GateStatus(
                        gate="C",
                        status=GateState.COMPLETED,
                        evidence=("checkpoint", "adapter", "evaluation", "peft_rag_smoke"),
                    )
            except (OSError, ValueError, TypeError):
                pass
        return (
            GateStatus(
                gate="A",
                status=GateState.PASSED if ready else GateState.PENDING,
                evidence=("calibrated_retrieval", "extractive_only") if ready else (),
                reason=None if ready else "calibrated retrieval artifacts are unavailable",
            ),
            gate_b,
            gate_c,
        )

    def _index(self) -> BM25Index:
        try:
            return load_index(self.settings)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            raise ServiceUnavailable("calibrated local retrieval is unavailable") from error

    def _models(self, index: BM25Index) -> tuple[DenseEncoder | None, Reranker | None]:
        if index.mode == "bm25_only":
            return None, None
        if self._runtime_fingerprint != index.fingerprint:
            try:
                self._embedder, self._reranker = load_runtime_models(self.settings, index)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                raise ServiceUnavailable("locked retrieval models are unavailable") from error
            self._runtime_fingerprint = index.fingerprint
        return self._embedder, self._reranker

    @staticmethod
    def _warning(
        index: BM25Index,
    ) -> Literal["bm25_only_fallback", "mvp_uncalibrated_retrieval"] | None:
        if index.mode == "bm25_only":
            return "bm25_only_fallback"
        if index.mode == "mvp_hybrid":
            return "mvp_uncalibrated_retrieval"
        return None

    def health(self, *, live_only: bool = False) -> HealthResponse:
        ready = is_ready(self.settings)
        mode: RetrievalMode = "bm25_only"
        if ready:
            try:
                mode = load_index(self.settings).mode
            except (OSError, ValueError, KeyError, RuntimeError):
                ready = False
        return HealthResponse(
            status="live" if live_only else ("ready" if ready else "not_ready"),
            retrieval_mode=mode,
            ready=ready,
            gates=self.gate_statuses(),
        )

    def _abstention(
        self,
        index: BM25Index,
    ) -> ChatResponse:
        return ChatResponse(
            answer=self.policy.abstention_response,
            citations=(),
            scores=(),
            mode=index.mode,
            warning=self._warning(index),
            outcome="abstained",
        )

    def _try_generated(
        self, request: ChatRequest, contexts: tuple[ProviderContext, ...]
    ) -> tuple[str, tuple[Citation, ...]] | None:
        if not self.generative_ready or self.generative_provider is None:
            return None
        try:
            calibration = load_support_calibration(self.settings)
            if self._support_scorer is None:
                self._support_scorer = TransformersClaimScorer(
                    self.settings.workspace, self.settings.models["support_verifier"]
                )
            generation = self.settings.rag["generation"]
            budget = PromptBudget(
                context_window_tokens=min(
                    int(generation["context_window_tokens"]),
                    int(getattr(self.generative_provider, "context_window_tokens", 32768)),
                ),
                output_tokens=int(generation["output_tokens"]),
                maximum_prompt_bytes=int(generation["maximum_prompt_bytes"]),
            )
            prompt = render_grounded_prompt(
                self.settings.workspace / "prompts" / "grounded_v1.txt",
                request.message,
                request.history,
                contexts,
                budget,
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            return None
        if not prompt.contexts:
            return None
        deadline = time.monotonic() + float(generation["deadline_seconds"])
        attempts = min(2, max(1, int(generation["maximum_attempts"])))
        for attempt in range(attempts):
            active_prompt = prompt if attempt == 0 else correction_prompt(prompt)
            try:
                result = self.generative_provider.generate(
                    ProviderRequest(active_prompt, deadline)
                )
                citations = validate_generated_result(
                    result,
                    prompt,
                    self.policy,
                    self._support_scorer,
                    calibration.threshold,
                )
                return result.text, citations
            except (GroundingError, ProviderError, OSError, ValueError, RuntimeError) as error:
                # No question, context or candidate text is emitted into logs.
                reason = str(error) if isinstance(error, GroundingError) else type(error).__name__
                print(json.dumps({"event": "generation_rejected", "attempt": attempt + 1,
                                  "reason": reason}), flush=True)
                continue
        return None

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.last_retrieval_hits = ()
        self.last_failure_reason = None
        decision = self.policy.classify_query(request.message)
        if not decision.retrieves:
            self.last_failure_reason = f"policy:{decision.action.value}"
            outcome: Literal["urgent", "refused"]
            if decision.action in {SafetyAction.URGENT, SafetyAction.SELF_HARM}:
                outcome = "urgent"
            else:
                outcome = "refused"
            return ChatResponse(
                answer=decision.response or self.policy.refusal_response,
                citations=(),
                scores=(),
                outcome=outcome,
            )
        index = self._index()
        try:
            calibration = load_calibration(self.settings, index)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            raise ServiceUnavailable("calibrated local retrieval is unavailable") from error
        embedder, reranker = self._models(index)
        hits = retrieve(
            index,
            request.message,
            calibration.thresholds,
            max(
                int(self.settings.rag["bm25"]["top_k"]),
                int(self.settings.rag["hybrid"]["maximum_contexts"]),
            ),
            embedder=embedder,
            reranker=reranker,
            candidate_k=int(self.settings.rag["bm25"]["candidate_k"]),
        )
        self.last_retrieval_hits = hits
        supported = tuple(
            hit for hit in hits if hit.supported and self.policy.role_allowed(hit.document.role)
        )[: int(self.settings.rag["hybrid"]["maximum_contexts"])]
        if not supported:
            self.last_failure_reason = "no_supported_context"
            return self._abstention(index)
        contexts = tuple(
            ProviderContext(
                chunk_id=hit.document.chunk_id,
                source=hit.document.source,
                page=hit.document.page,
                question=hit.document.question,
                text=hit.document.text,
                role=hit.document.role,
            )
            for hit in supported
        )
        generated = self._try_generated(request, contexts)
        if generated is None:
            result = self.provider.answer(request.message, contexts)
            if result is None:
                self.last_failure_reason = "no_matching_extractive_sentence"
                return self._abstention(index)
            try:
                citations = validate_extractive_result(result, contexts, self.policy)
            except GroundingError as error:
                self.last_failure_reason = f"extractive_validation:{error}"
                return self._abstention(index)
            answer = result.text
        else:
            answer, citations = generated
        by_id = {hit.document.chunk_id: hit for hit in supported}
        scores = tuple(
            RetrievalScores(
                bm25=by_id[citation.chunk_id].bm25_score,
                dense=by_id[citation.chunk_id].dense_score,
                rerank=by_id[citation.chunk_id].rerank_score,
            )
            for citation in citations
        )
        return ChatResponse(
            answer=answer,
            citations=citations,
            scores=scores,
            mode=index.mode,
            warning=self._warning(index),
            outcome="answered",
        )
