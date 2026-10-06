from __future__ import annotations

import json
from pathlib import Path

import pytest

from acharya.config import Settings
from acharya.rag.evaluate import calibrate, is_ready, load_calibration
from acharya.rag.index import load_index


def test_calibration_has_zero_false_support_and_matching_hashes(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    result = load_calibration(settings)
    assert result.ready
    assert result.false_support == 0
    assert result.thresholds.raw_score != 0.75


def test_missing_and_stale_calibration_refuse_readiness(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    path = settings.state_dir / "bm25_calibration.json"
    value = json.loads(path.read_text())
    value["index_fingerprint"] = "stale"
    path.write_text(json.dumps(value))
    assert not is_ready(settings)
    with pytest.raises(RuntimeError, match="stale"):
        load_calibration(settings)
    calibrate(settings, load_index(settings))
    assert is_ready(settings)
    evaluation = settings.workspace / "eval" / "golden.jsonl"
    evaluation.write_text(evaluation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not is_ready(settings)
    with pytest.raises(RuntimeError, match="stale"):
        load_calibration(settings)


def test_calibration_runs_neural_retrieval_once_per_unique_query(
    built_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from acharya.rag.index import load_index
    from acharya.rag.retrieve import retrieve as real_retrieve

    settings = Settings.load(built_workspace)
    base = load_index(settings)
    index = replace(base, mode="full",
                    dense_vectors=tuple((1.0, 0.0) for _ in base.documents))
    calls: list[str] = []

    class Encoder:
        def encode_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

        def encode_passages(self, passages: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in passages]

    class Reranker:
        def score(self, query: str, passages: list[str]) -> list[float]:
            calls.append(query)
            return [0.99 for _ in passages]

    monkeypatch.setattr("acharya.rag.evaluate.retrieve", real_retrieve)
    calibrate(settings, index, embedder=Encoder(), reranker=Reranker())
    assert calls
    assert len(calls) == len(set(calls))


def test_joint_thresholds_can_reject_high_bm25_negative_using_neural_scores() -> None:
    from acharya.rag.evaluate import _joint_candidates
    from acharya.rag.index import BM25Document
    from acharya.rag.retrieve import RetrievalHit, threshold_support

    document = BM25Document('id', 'source', 1, '', 'evidence', 'primary')
    positive = RetrievalHit(document, 3.0, 5.0, 0.3, 2, False, 0.6, 0.15)
    negative = RetrievalHit(document, 30.0, 5.0, 0.9, 2, False, 0.7, 0.01)
    assert any(threshold_support(positive, t, 'full', 4)
               and not threshold_support(negative, t, 'full', 4)
               for t in _joint_candidates([positive, negative]))


def test_score_cache_reuses_queries_and_invalidates_changed_evaluation(
    built_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import acharya.rag.evaluate as module

    settings = Settings.load(built_workspace)
    original = module.retrieve
    calls = []

    def recording(*args, **kwargs):
        if kwargs.get('top_k') != 1:
            calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'retrieve', recording)
    calibrate(settings)
    assert not calls  # built_workspace already calibrated and persisted scores
    path = settings.workspace / 'eval/golden.jsonl'
    path.write_text(path.read_text() + '\n')
    calibrate(settings)
    assert calls
    calls.clear()
    calibrate(settings)
    assert not calls
    for cache in (settings.state_dir / 'retrieval_scores').rglob('*.json'):
        cache.write_text('{}')
    calibrate(settings)
    assert calls
