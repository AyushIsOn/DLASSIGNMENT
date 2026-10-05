from __future__ import annotations

from pathlib import Path

import pytest

from acharya.providers.base import ProviderContext, ProviderResult
from acharya.rag.grounding import (
    GroundingError,
    parse_generated_claims,
    validate_extractive_result,
    validate_generated_result,
)
from acharya.rag.prompt import RenderedPrompt
from acharya.safety import SafetyPolicy


def test_citation_maps_one_to_one_and_requires_extract(project_root: Path) -> None:
    policy = SafetyPolicy.load(project_root / "configs" / "safety.yaml")
    context = ProviderContext("one", "pdf", 2, "question", "Vata is described here.", "educational")
    result = ProviderResult("Vata is described here.", ("one",), "extractive")
    citations = validate_extractive_result(result, (context,), policy)
    assert len(citations) == 1
    assert citations[0].chunk_id == "one"
    with pytest.raises(GroundingError, match="not extractive"):
        validate_extractive_result(
            ProviderResult("Unsupported assertion.", ("one",), "extractive"), (context,), policy
        )


def test_unsafe_candidate_rejected_whole(project_root: Path) -> None:
    policy = SafetyPolicy.load(project_root / "configs" / "safety.yaml")
    text = "You should take 5 mg."
    context = ProviderContext("one", "pdf", 2, "question", text, "educational")
    with pytest.raises(GroundingError, match="safety"):
        validate_extractive_result(ProviderResult(text, ("one",), "extractive"), (context,), policy)


class FixedScorer:
    def __init__(self, score: float) -> None:
        self.value = score
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def score(self, premises: tuple[str, ...], hypothesis: str) -> float:
        self.calls.append((premises, hypothesis))
        return self.value


def test_generated_claims_require_cited_entailment_and_reject_whole(project_root: Path) -> None:
    policy = SafetyPolicy.load(project_root / "configs" / "safety.yaml")
    first = ProviderContext("one", "pdf", 2, "q", "Vata is a dosha.", "educational")
    second = ProviderContext("two", "pdf", 3, "q", "Kapha is a dosha.", "educational")
    prompt = RenderedPrompt("p", b"p", 1, 8, (first, second))
    scorer = FixedScorer(0.9)
    result = ProviderResult("Vata is a dosha [1].\n- Kapha is a dosha [2].", (), "fake")
    citations = validate_generated_result(result, prompt, policy, scorer, 0.8)
    assert tuple(item.chunk_id for item in citations) == ("one", "two")
    assert scorer.calls[0][0] == (first.text,)
    with pytest.raises(GroundingError, match="unsupported"):
        validate_generated_result(result, prompt, policy, FixedScorer(0.1), 0.8)
    with pytest.raises(GroundingError, match="citation"):
        parse_generated_claims("Unsupported sentence without evidence.")
    with pytest.raises(GroundingError, match="malformed"):
        parse_generated_claims("| claim | citation |\n| --- | --- |")


def test_paragraph_citation_scopes_each_claim_without_crossing_paragraphs() -> None:
    claims = parse_generated_claims("Vata concerns movement. Kapha concerns nourishment. [1]")
    assert len(claims) == 2
    assert all(claim.citation_numbers == (1,) for claim in claims)
    with pytest.raises(GroundingError, match="lacks a citation"):
        parse_generated_claims("Uncited claim.\n\nCited claim [1]")


def test_inherited_citations_still_require_entailment_for_every_claim(project_root: Path) -> None:
    policy = SafetyPolicy.load(project_root / "configs/safety.yaml")
    context = ProviderContext("one", "fixture", 1, "q", "Vata concerns movement.", "educational")
    prompt = RenderedPrompt("p", b"p", 1, 64, (context,))

    class SelectiveScorer:
        def score(self, premises: tuple[str, ...], hypothesis: str) -> float:
            return 0.99 if hypothesis == "Vata concerns movement." else 0.0

    result = ProviderResult("Vata concerns movement. Kapha cures diabetes. [1]", (), "fake")
    with pytest.raises(GroundingError):
        validate_generated_result(result, prompt, policy, SelectiveScorer(), 0.8)
