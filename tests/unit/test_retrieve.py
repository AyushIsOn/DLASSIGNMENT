from __future__ import annotations

from acharya.rag.index import BM25Document, BM25Index
from acharya.rag.retrieve import SupportThresholds, retrieve


def _index() -> BM25Index:
    documents = (
        BM25Document("a", "fixture", 1, "", "alpha beta", "educational"),
        BM25Document("b", "fixture", 1, "", "alpha beta", "educational"),
    )
    return BM25Index(
        "fingerprint",
        "corpus",
        "config",
        documents,
        (("alpha", "beta"), ("alpha", "beta")),
        {"alpha": 1.0, "beta": 1.0, "incidental": 1.0},
        2.0,
        1.5,
        0.75,
    )


def test_equal_scores_sort_by_chunk_id() -> None:
    assert [hit.document.chunk_id for hit in retrieve(_index(), "alpha beta")] == ["a", "b"]


def test_support_requires_distinct_informative_overlap_and_raw_floor() -> None:
    thresholds = SupportThresholds(0.5, 0.1, 2)
    assert retrieve(_index(), "alpha incidental", thresholds)[0].supported is False
    assert retrieve(_index(), "alpha beta", thresholds)[0].supported is True
    high = SupportThresholds(0.5, 100.0, 2)
    assert retrieve(_index(), "alpha beta", high)[0].supported is False


class _DenseNearEncoder:
    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in passages]

    def encode_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class _HighReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.99 for _ in passages]


def test_dense_near_context_cannot_bypass_lexical_support() -> None:
    base = _index()
    thresholds = SupportThresholds(0.5, 0.1, 2, dense_score=0.5, rerank_score=0.5)
    encoder = _DenseNearEncoder()
    for mode in ("mvp_hybrid", "full"):
        index = BM25Index(
            base.fingerprint,
            base.corpus_fingerprint,
            base.config_hash,
            base.documents,
            base.document_tokens,
            base.idf,
            base.average_length,
            base.k1,
            base.b,
            mode,  # type: ignore[arg-type]
            ((1.0, 0.0), (1.0, 0.0)),
        )
        hits = retrieve(
            index,
            "incidental",
            thresholds,
            embedder=encoder,
            reranker=_HighReranker() if mode == "full" else None,
        )
        assert hits[0].dense_score == 1.0
        assert hits[0].supported is False


def test_full_retrieval_accepts_complementary_evidence_routes() -> None:
    base = _index()
    index = BM25Index(
        base.fingerprint,
        base.corpus_fingerprint,
        base.config_hash,
        base.documents,
        base.document_tokens,
        base.idf,
        base.average_length,
        base.k1,
        base.b,
        "full",
        ((1.0, 0.0), (1.0, 0.0)),
    )
    # Semantic evidence compensates for a query modifier absent from the passage.
    thresholds = SupportThresholds(0.8, 0.1, 2, dense_score=0.7, rerank_score=0.8)
    hits = retrieve(
        index,
        "alpha beta incidental",
        thresholds,
        embedder=_DenseNearEncoder(),
        reranker=_HighReranker(),
    )
    assert hits[0].lexical_coverage < 0.8
    assert hits[0].supported
    # Strong lexical evidence can pass even when dense similarity misses its floor.
    dense_floor = SupportThresholds(0.8, 0.1, 2, dense_score=1.01, rerank_score=0.8)
    assert retrieve(
        index, "alpha beta", dense_floor, embedder=_DenseNearEncoder(), reranker=_HighReranker()
    )[0].supported
    # Neither route bypasses the BM25 anchor or reranker.
    for blocked in (
        SupportThresholds(0.8, 100, 2, 0.7, 0.8),
        SupportThresholds(0.8, 0.1, 2, 0.7, 1.0),
    ):
        assert not retrieve(
            index, "alpha beta", blocked, embedder=_DenseNearEncoder(), reranker=_HighReranker()
        )[0].supported


def test_small_source_is_reranked_even_outside_global_candidate_cutoff() -> None:
    from dataclasses import replace

    base = _index()
    documents = (
        *base.documents,
        BM25Document("c", "small-source", 1, "A different question", "alpha", "educational"),
    )
    index = replace(
        base,
        documents=documents,
        document_tokens=(*base.document_tokens, ("alpha",)),
        mode="full",
        dense_vectors=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    )

    class RecordingReranker:
        def __init__(self) -> None:
            self.passages: list[str] = []

        def score(self, query: str, passages: list[str]) -> list[float]:
            self.passages = passages
            return [0.99 for _ in passages]

    reranker = RecordingReranker()
    retrieve(index, "alpha beta", candidate_k=1, embedder=_DenseNearEncoder(), reranker=reranker)
    assert any("A different question" in passage for passage in reranker.passages)
