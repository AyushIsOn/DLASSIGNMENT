"""Citation identity and whole-candidate claim-support validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from acharya.providers.base import ProviderContext, ProviderResult
from acharya.rag.embed import acquire_locked_model
from acharya.rag.prompt import RenderedPrompt
from acharya.safety import SafetyPolicy
from acharya.schemas import Citation

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


class GroundingError(ValueError):
    """Raised when a candidate cannot be proven against its contexts."""


class ClaimScorer(Protocol):
    def score(self, premises: tuple[str, ...], hypothesis: str) -> float: ...


@dataclass(frozen=True)
class GeneratedClaim:
    hypothesis: str
    citation_numbers: tuple[int, ...]


def validate_extractive_result(
    result: ProviderResult,
    contexts: tuple[ProviderContext, ...],
    policy: SafetyPolicy,
) -> tuple[Citation, ...]:
    context_by_id = {item.chunk_id: item for item in contexts}
    if not result.text or not policy.candidate_allowed(result.text):
        raise GroundingError("candidate rejected by safety policy")
    citations: list[Citation] = []
    for chunk_id in result.used_chunk_ids:
        context = context_by_id.get(chunk_id)
        if context is None or not policy.role_allowed(context.role):
            raise GroundingError("candidate cited an unavailable or disallowed context")
        citations.append(_citation(context))
    if not citations:
        raise GroundingError("candidate has no citations")
    candidate_sentences = tuple(
        sentence.strip() for sentence in _SENTENCE.split(result.text) if sentence.strip()
    )
    if not candidate_sentences or not all(
        any(sentence in context.text for context in contexts) for sentence in candidate_sentences
    ):
        raise GroundingError("candidate is not extractive")
    return tuple(citations)


def _citation(context: ProviderContext) -> Citation:
    return Citation(
        chunk_id=context.chunk_id,
        source=context.source,
        page=context.page,
        text=context.text,
    )


def parse_generated_claims(text: str) -> tuple[GeneratedClaim, ...]:
    if not text.strip() or "```" in text or any("|" in line for line in text.splitlines()):
        raise GroundingError("candidate structure is malformed")
    claims: list[GeneratedClaim] = []
    for line in text.splitlines():
        clean_line = _LIST_PREFIX.sub("", line.strip())
        if not clean_line:
            continue
        for sentence in _SENTENCE.split(clean_line):
            sentence = sentence.strip()
            if not sentence:
                continue
            citations = tuple(int(item) for item in _CITATION.findall(sentence))
            hypothesis = _CITATION.sub("", sentence).strip()
            hypothesis = re.sub(r"\s+([.!?])", r"\1", hypothesis).strip()
            if not citations or not re.search(r"[A-Za-z]", hypothesis):
                raise GroundingError("substantive claim lacks a citation")
            if "[" in hypothesis or "]" in hypothesis or any(number < 1 for number in citations):
                raise GroundingError("candidate citation is malformed")
            claims.append(GeneratedClaim(hypothesis, tuple(dict.fromkeys(citations))))
    if not claims:
        raise GroundingError("candidate has no substantive claims")
    return tuple(claims)


class TransformersClaimScorer:
    """Pinned 512-token NLI scoring with premise-only overlap windows."""

    def __init__(self, workspace: Path, lock: dict[str, Any]) -> None:
        expected = {str(key): str(value) for key, value in lock.get("files", {}).items()}
        if not expected:
            raise RuntimeError("support verifier file hashes are not locked")
        model = acquire_locked_model(workspace, lock)
        if model.files != expected:
            raise RuntimeError("support verifier files do not match the lock")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model.path), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model.path), local_files_only=True, trust_remote_code=False
        )
        self.model.eval()

    def score(self, premises: tuple[str, ...], hypothesis: str) -> float:
        import torch

        hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        special = int(self.tokenizer.num_special_tokens_to_add(pair=True))
        capacity = 512 - len(hypothesis_ids) - special
        if not hypothesis_ids or capacity <= 0:
            raise GroundingError("claim hypothesis exceeds verifier budget")
        step = max(1, capacity - 64)
        best = 0.0
        for premise in premises:
            premise_ids = self.tokenizer.encode(premise, add_special_tokens=False)
            starts = range(0, max(1, len(premise_ids)), step)
            for start in starts:
                window = premise_ids[start : start + capacity]
                if not window:
                    continue
                input_ids = self.tokenizer.build_inputs_with_special_tokens(
                    window, hypothesis_ids
                )
                token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                    window, hypothesis_ids
                )
                encoded = {
                    "input_ids": torch.tensor([input_ids]),
                    "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                    "token_type_ids": torch.tensor([token_type_ids]),
                }
                with torch.no_grad():
                    logits = self.model(**encoded).logits[0]
                probabilities = torch.softmax(logits, dim=-1).tolist()
                if len(probabilities) < 3:
                    raise RuntimeError("support verifier has an invalid label space")
                entailment = float(probabilities[0])
                if entailment > float(probabilities[1]) and entailment > float(probabilities[2]):
                    best = max(best, entailment)
                if start + capacity >= len(premise_ids):
                    break
        return best


def validate_generated_result(
    result: ProviderResult,
    prompt: RenderedPrompt,
    policy: SafetyPolicy,
    scorer: ClaimScorer,
    threshold: float,
) -> tuple[Citation, ...]:
    if not result.text or not policy.candidate_allowed(result.text):
        raise GroundingError("candidate rejected by safety policy")
    if not math.isfinite(threshold) or not 0.5 <= threshold <= 0.99:
        raise GroundingError("support verifier threshold is invalid")
    claims = parse_generated_claims(result.text)
    used: list[int] = []
    for claim in claims:
        try:
            cited = tuple(prompt.contexts[number - 1] for number in claim.citation_numbers)
        except IndexError as error:
            raise GroundingError("candidate cited an unavailable context") from error
        if not cited or any(not policy.role_allowed(context.role) for context in cited):
            raise GroundingError("candidate cited an unavailable or disallowed context")
        if scorer.score(tuple(context.text for context in cited), claim.hypothesis) < threshold:
            raise GroundingError("candidate contains an unsupported claim")
        for number in claim.citation_numbers:
            if number not in used:
                used.append(number)
    return tuple(_citation(prompt.contexts[number - 1]) for number in used)
