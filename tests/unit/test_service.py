from __future__ import annotations

from pathlib import Path

from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest, GateState


def test_service_answers_with_citation_and_null_optional_scores(built_workspace: Path) -> None:
    service = RAGService(built_workspace)
    response = service.chat(ChatRequest(message="Which are the doshas involved in Eka kusta?"))
    assert response.outcome == "answered"
    assert response.citations
    assert response.warning == "bm25_only_fallback"
    assert all(item.dense is None and item.rerank is None for item in response.scores)


def test_safety_bypasses_retrieval_and_unsupported_abstains(built_workspace: Path) -> None:
    service = RAGService(built_workspace)
    urgent = service.chat(ChatRequest(message="I cannot breathe."))
    assert urgent.outcome == "urgent"
    assert urgent.citations == ()
    unsupported = service.chat(ChatRequest(message="How do I repair a diesel engine gearbox?"))
    assert unsupported.outcome == "abstained"


def test_gate_statuses_are_independent(built_workspace: Path) -> None:
    gates = RAGService(built_workspace).gate_statuses()
    assert gates[0].status is GateState.PASSED
    assert gates[1].status is GateState.PENDING
    assert gates[2].status is GateState.PENDING_EXTERNAL_GPU
