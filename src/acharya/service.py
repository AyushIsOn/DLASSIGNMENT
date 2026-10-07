"""Chat orchestration: safety routing -> BM25 retrieval -> fine-tuned model -> output guard."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from acharya.generation import GenerationError, Generator
from acharya.prompting import Message, build_messages, render_prompt
from acharya.retrieval import BM25, Hit
from acharya.safety import Action, SafetyPolicy
from acharya.schemas import ChatMessage, ChatRequest, ChatResponse, Citation, RetrievalScores

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB = ROOT / "data" / "kb" / "cards.jsonl"
DEFAULT_SAFETY = ROOT / "configs" / "safety.yaml"


class ModelUnavailable(RuntimeError):
    """The generation backend could not answer (mapped to HTTP 503)."""


class ChatService:
    def __init__(
        self,
        generator: Generator,
        *,
        kb_path: Path = DEFAULT_KB,
        safety_path: Path = DEFAULT_SAFETY,
        use_retrieval: bool = True,
        top_k: int = 3,
        min_score: float = 7.5,  # keeps ~97% of real KB questions, drops small talk
        max_new_tokens: int = 384,
        history_pairs: int = 2,
    ) -> None:
        self.generator = generator
        self.index = BM25.from_jsonl(kb_path)
        self.policy = SafetyPolicy.load(safety_path)
        self.use_retrieval = use_retrieval
        self.top_k = top_k
        self.min_score = min_score
        self.max_new_tokens = max_new_tokens
        self.history_pairs = history_pairs

    def retrieve(self, question: str, history: Sequence[ChatMessage] = ()) -> list[Hit]:
        """Top-k cards (like training) only when the best match is a confident one.
        Follow-ups ("Is it curable?") are searched together with the previous question."""
        if not self.use_retrieval:
            return []
        previous = [item.content for item in history if item.role == "user"][-1:]
        hits = self.index.search(" ".join([question, *previous]), k=self.top_k)
        if not hits or hits[0].score < self.min_score:
            return []
        return hits

    def _history(self, history: Sequence[ChatMessage]) -> list[Message]:
        keep = list(history)[-2 * self.history_pairs:] if self.history_pairs else []
        return [{"role": item.role, "content": item.content} for item in keep]

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        decision = self.policy.classify(request.message)
        if decision.action is not Action.ANSWER:
            outcome = "urgent" if decision.action in {Action.URGENT, Action.SELF_HARM} \
                else "refused"
            return ChatResponse(answer=decision.response or "", mode="safety", outcome=outcome,
                                latency_ms=int(1000 * (time.monotonic() - started)))
        hits = self.retrieve(request.message, request.history)
        messages = build_messages(request.message, [hit.card.text for hit in hits],
                                  self._history(request.history))
        try:
            answer = self.generator.generate(render_prompt(messages), self.max_new_tokens)
        except GenerationError as error:
            raise ModelUnavailable(str(error)) from error
        if not answer.strip():
            raise ModelUnavailable("the model returned an empty answer")
        answer, replaced = self.policy.guard_answer(answer)
        shown = [hit for hit in hits if hit.score >= 0.6 * hits[0].score] if hits else []
        return ChatResponse(
            answer=answer,
            citations=tuple(Citation(chunk_id=hit.card.id, source=hit.card.source, page=0,
                                     text=hit.card.text) for hit in shown),
            scores=tuple(RetrievalScores(bm25=round(hit.score, 3)) for hit in shown),
            mode="finetuned_rag" if hits else "finetuned",
            warning="dose_removed" if replaced else None,
            outcome="refused" if replaced else "answered",
            model=self.generator.name,
            latency_ms=int(1000 * (time.monotonic() - started)),
        )
