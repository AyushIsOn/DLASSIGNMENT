from __future__ import annotations

import json
from pathlib import Path

from acharya.config import Settings
from acharya.ingest.pipeline import build_corpus
from acharya.rag.evaluate import calibrate, load_calibration
from acharya.rag.index import build_index


def test_answerable_miss_remains_in_calibration_denominator(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    build_corpus(settings)
    index = build_index(settings, "bm25_only")
    golden = root / "eval" / "golden.jsonl"
    initial = calibrate(settings, index, golden=golden, activate=False)
    with golden.open("a") as handle:
        handle.write(json.dumps({"id": "known-miss", "kind": "answerable",
                                 "query": "xqvzz absententity"}) + "\n")
    result = calibrate(settings, index, golden=golden, activate=False)
    assert result.answerable_count == initial.answerable_count + 1
    assert result.answerable_supported == initial.answerable_supported


class FakeEncoder:
    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in passages]

    def encode_query(self, query: str) -> list[float]:
        adversarial = ("zxqv", "martian", "diesel")
        return [0.0, 1.0] if any(term in query.casefold() for term in adversarial) else [1.0, 0.0]


class OverlapReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9 for _ in passages]


def test_full_calibration_is_model_bound_and_has_zero_false_support(
    workspace_factory: object,
) -> None:
    root: Path = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    build_corpus(settings)
    encoder = FakeEncoder()
    reranker = OverlapReranker()
    index = build_index(settings, "full", embedder=encoder, reranker=reranker)
    golden = root / "eval" / "golden.jsonl"
    result = calibrate(
        settings,
        index,
        golden=golden,
        embedder=encoder,
        reranker=reranker,
    )
    assert result.ready
    assert result.false_support == 0
    assert result.model_hashes == index.model_hashes
    assert result.thresholds.dense_score is not None
    assert result.thresholds.rerank_score is not None
    assert set(result.adversarial_categories) == {
        "typo",
        "entity_swap",
        "negation",
        "random",
        "irrelevant",
    }
    assert load_calibration(settings, index) == result
    golden.write_text(golden.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        load_calibration(settings, index)
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("changed full evaluation source remained ready")


def test_hybrid_calibration_rejects_changed_evaluation_source(
    workspace_factory: object,
) -> None:
    root: Path = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    build_corpus(settings)
    encoder = FakeEncoder()
    index = build_index(settings, "mvp_hybrid", embedder=encoder)
    golden = root / "eval" / "golden.jsonl"
    calibrate(settings, index, golden=golden, embedder=encoder)
    assert load_calibration(settings, index).retrieval_mode == "mvp_hybrid"
    golden.write_text(golden.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        load_calibration(settings, index)
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("changed hybrid evaluation source remained ready")
