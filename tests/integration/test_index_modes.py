from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.ingest.pipeline import build_corpus
from acharya.rag.index import build_index, load_index
from acharya.rag.retrieve import SupportThresholds, retrieve


class FakeEncoder:
    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        return [[float(index + 1), 1.0] for index, _ in enumerate(passages)]

    def encode_query(self, query: str) -> list[float]:
        return [1.0, 1.0]


class FakeReranker:
    def __init__(self) -> None:
        self.count = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.count = len(passages)
        return [float(index) / max(len(passages), 1) for index, _ in enumerate(passages)]


def test_modes_store_nullable_scores_and_rerank_entire_union(workspace_factory: object) -> None:
    root: Path = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    build_corpus(settings)
    bm25 = build_index(settings)
    assert bm25.mode == "bm25_only" and not bm25.dense_vectors
    encoder = FakeEncoder()
    hybrid = build_index(settings, "mvp_hybrid", embedder=encoder)
    assert load_index(settings, "mvp_hybrid").fingerprint == hybrid.fingerprint
    hybrid_hits = retrieve(
        hybrid,
        "Eka kusta dosha",
        SupportThresholds(0.0, -100.0, 1, -1.0),
        top_k=40,
        embedder=encoder,
    )
    assert hybrid_hits and all(
        item.dense_score is not None and item.rerank_score is None for item in hybrid_hits
    )
    reranker = FakeReranker()
    full = build_index(settings, "full", embedder=encoder, reranker=reranker)
    full_hits = retrieve(
        full,
        "Eka kusta dosha",
        SupportThresholds(0.0, -100.0, 1, -1.0, -1.0),
        top_k=40,
        embedder=encoder,
        reranker=reranker,
    )
    assert reranker.count == len(full_hits)
    assert all(item.dense_score is not None and item.rerank_score is not None for item in full_hits)
