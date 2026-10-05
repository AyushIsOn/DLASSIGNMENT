"""Raw lexical support, dense/BM25 union, RRF, and all-candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass

from acharya.rag.embed import DenseEncoder, Reranker, cosine
from acharya.rag.index import BM25Document, BM25Index, retrieval_passage, tokenize


@dataclass(frozen=True)
class RetrievalHit:
    document: BM25Document
    bm25_score: float
    positive_query_idf: float
    lexical_coverage: float
    informative_overlap: int
    supported: bool
    dense_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass(frozen=True)
class SupportThresholds:
    lexical_coverage: float
    raw_score: float
    minimum_overlap: int
    dense_score: float | None = None
    rerank_score: float | None = None


def _lexical_supported(
    positive_idf: float,
    overlap: int,
    coverage: float,
    score: float,
    informative_count: int,
    thresholds: SupportThresholds | None,
) -> bool:
    if thresholds is None:
        return False
    required = 1 if informative_count == 1 else thresholds.minimum_overlap
    return (
        positive_idf > 0
        and overlap >= required
        and coverage >= thresholds.lexical_coverage
        and score >= thresholds.raw_score
    )


def retrieve(
    index: BM25Index,
    query: str,
    thresholds: SupportThresholds | None = None,
    top_k: int = 5,
    *,
    embedder: DenseEncoder | None = None,
    reranker: Reranker | None = None,
    candidate_k: int = 20,
) -> tuple[RetrievalHit, ...]:
    query_tokens = tokenize(query)
    informative = tuple(sorted({term for term in query_tokens if index.idf.get(term, 0.0) > 0.0}))
    positive_query_idf = sum(index.idf[term] for term in informative)
    lexical: list[tuple[int, float, float, int, bool]] = []
    for position, terms in enumerate(index.document_tokens):
        document_terms = set(terms)
        overlapping = tuple(term for term in informative if term in document_terms)
        overlap_mass = sum(index.idf[term] for term in overlapping)
        coverage = overlap_mass / positive_query_idf if positive_query_idf > 0 else 0.0
        score = index.score(query_tokens, position)
        supported = _lexical_supported(
            positive_query_idf, len(overlapping), coverage, score, len(informative), thresholds
        )
        lexical.append((position, score, coverage, len(overlapping), supported))
    bm25_order = sorted(lexical, key=lambda item: (-item[1], index.documents[item[0]].chunk_id))
    bm25_ranks = {item[0]: rank for rank, item in enumerate(bm25_order[:candidate_k], start=1)}
    if index.mode == "bm25_only":
        hits = [
            RetrievalHit(
                index.documents[position],
                score,
                positive_query_idf,
                coverage,
                overlap,
                supported,
                rrf_score=1.0 / (60 + rank),
                bm25_rank=rank,
            )
            for rank, (position, score, coverage, overlap, supported) in enumerate(
                bm25_order, start=1
            )
        ]
        return tuple(hits[:top_k])
    if embedder is None:
        raise RuntimeError("dense encoder is required for hybrid retrieval")
    query_vector = embedder.encode_query(query)
    dense_scores = [cosine(query_vector, list(vector)) for vector in index.dense_vectors]
    dense_order = sorted(
        range(len(index.documents)),
        key=lambda position: (-dense_scores[position], index.documents[position].chunk_id),
    )
    dense_ranks = {
        position: rank for rank, position in enumerate(dense_order[:candidate_k], start=1)
    }
    # Large homogeneous sources must not exclude every candidate from smaller ones.
    # Add bounded per-source lexical/dense candidates without changing support gates.
    diverse: set[int] = set()
    for quota in (1, 2):
        for ordered in ([item[0] for item in bm25_order], dense_order):
            counts: dict[str, int] = {}
            for position in ordered:
                source = index.documents[position].source
                if counts.get(source, 0) < quota and len(diverse) < 2 * candidate_k:
                    diverse.add(position)
                    counts[source] = counts.get(source, 0) + 1
    union = sorted(
        set(bm25_ranks) | set(dense_ranks) | diverse,
        key=lambda position: index.documents[position].chunk_id,
    )
    hits = []
    for position in union:
        _, score, coverage, overlap, lexical_support = lexical[position]
        dense_score = dense_scores[position]
        # Dense similarity and lexical coverage are alternative evidence routes.
        # Both still require a lexical anchor and raw BM25 floor in full mode.
        anchor = (
            thresholds is not None
            and positive_query_idf > 0
            and overlap >= (1 if len(informative) == 1 else thresholds.minimum_overlap)
            and score >= thresholds.raw_score
        )
        dense_support = (
            thresholds is not None
            and thresholds.dense_score is not None
            and dense_score >= thresholds.dense_score
        )
        rrf = (1.0 / (60 + bm25_ranks[position]) if position in bm25_ranks else 0.0) + (
            1.0 / (60 + dense_ranks[position]) if position in dense_ranks else 0.0
        )
        hits.append(
            RetrievalHit(
                index.documents[position],
                score,
                positive_query_idf,
                coverage,
                overlap,
                bool(
                    anchor and (lexical_support or dense_support)
                    if index.mode == "full"
                    else lexical_support and dense_support
                ),
                dense_score,
                None,
                rrf,
                bm25_ranks.get(position),
                dense_ranks.get(position),
            )
        )
    if index.mode == "full":
        if reranker is None:
            raise RuntimeError("reranker is required for full retrieval")
        rerank_scores = reranker.score(query, [retrieval_passage(item.document) for item in hits])
        if len(rerank_scores) != len(hits):
            raise RuntimeError("reranker score count mismatch")
        hits = [
            RetrievalHit(
                item.document,
                item.bm25_score,
                item.positive_query_idf,
                item.lexical_coverage,
                item.informative_overlap,
                item.supported
                and thresholds is not None
                and thresholds.rerank_score is not None
                and rerank >= thresholds.rerank_score,
                item.dense_score,
                rerank,
                item.rrf_score,
                item.bm25_rank,
                item.dense_rank,
            )
            for item, rerank in zip(hits, rerank_scores, strict=True)
        ]
        hits.sort(
            key=lambda item: (
                -(item.rerank_score or 0.0),
                -(item.rrf_score or 0.0),
                item.document.chunk_id,
            )
        )
    else:
        hits.sort(
            key=lambda item: (
                -(item.rrf_score or 0.0),
                -(item.dense_score or 0.0),
                -item.bm25_score,
                item.document.chunk_id,
            )
        )
    return tuple(hits[:top_k])


def threshold_support(
    hit: RetrievalHit, thresholds: SupportThresholds, mode: str, informative_count: int
) -> bool:
    """Apply threshold grids to cached, threshold-independent retrieval scores."""
    lexical = _lexical_supported(
        hit.positive_query_idf,
        hit.informative_overlap,
        hit.lexical_coverage,
        hit.bm25_score,
        informative_count,
        thresholds,
    )
    if mode == "bm25_only":
        return lexical
    dense = (
        hit.dense_score is not None
        and thresholds.dense_score is not None
        and hit.dense_score >= thresholds.dense_score
    )
    if mode != "full":
        return bool(lexical and dense)
    required = 1 if informative_count == 1 else thresholds.minimum_overlap
    anchor = (
        hit.positive_query_idf > 0
        and hit.informative_overlap >= required
        and hit.bm25_score >= thresholds.raw_score
    )
    return bool(
        anchor
        and (lexical or dense)
        and hit.rerank_score is not None
        and thresholds.rerank_score is not None
        and hit.rerank_score >= thresholds.rerank_score
    )
