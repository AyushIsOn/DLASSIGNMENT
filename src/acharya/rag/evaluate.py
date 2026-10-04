"""Corpus/model-bound retrieval calibration with zero false support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from acharya.config import Settings, canonical_json, sha256_bytes, sha256_file
from acharya.rag.embed import (
    BGEDenseEncoder,
    BGEReranker,
    DenseEncoder,
    Reranker,
    acquire_locked_model,
)
from acharya.rag.index import BM25Index, load_index
from acharya.rag.retrieve import SupportThresholds, retrieve

_REQUIRED_ADVERSARIAL_CATEGORIES = frozenset(
    {"typo", "entity_swap", "negation", "random", "irrelevant"}
)


@dataclass(frozen=True)
class Calibration:
    index_fingerprint: str
    corpus_fingerprint: str
    rag_config_hash: str
    evaluation_source: str
    evaluation_source_sha256: str
    evaluation_hash: str
    thresholds: SupportThresholds
    answerable_count: int
    answerable_supported: int
    adversarial_count: int
    adversarial_categories: tuple[str, ...]
    false_support: int
    retrieval_mode: str = "bm25_only"
    model_hashes: dict[str, str] | None = None

    @property
    def ready(self) -> bool:
        return self.answerable_supported > 0 and self.false_support == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "index_fingerprint": self.index_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "rag_config_hash": self.rag_config_hash,
            "evaluation_source": self.evaluation_source,
            "evaluation_source_sha256": self.evaluation_source_sha256,
            "evaluation_hash": self.evaluation_hash,
            "retrieval_mode": self.retrieval_mode,
            "model_hashes": self.model_hashes or {},
            "thresholds": {
                "lexical_coverage": self.thresholds.lexical_coverage,
                "raw_score": self.thresholds.raw_score,
                "minimum_overlap": self.thresholds.minimum_overlap,
                "dense_score": self.thresholds.dense_score,
                "rerank_score": self.thresholds.rerank_score,
            },
            "answerable_count": self.answerable_count,
            "answerable_supported": self.answerable_supported,
            "adversarial_count": self.adversarial_count,
            "adversarial_categories": list(self.adversarial_categories),
            "false_support": self.false_support,
            "ready": self.ready,
        }


def _evaluation_source(settings: Settings, golden: Path | None) -> tuple[Path, str]:
    path = (golden or settings.workspace / "eval" / "golden.jsonl").resolve()
    if not path.is_file():
        raise RuntimeError("retrieval evaluation source is missing")
    try:
        source = path.relative_to(settings.workspace).as_posix()
    except ValueError:
        source = str(path)
    return path, source


def _queries(
    settings: Settings, index: BM25Index, golden: Path | None
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, str, tuple[str, ...]]:
    path, source = _evaluation_source(settings, golden)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise RuntimeError("golden IDs must be nonempty and unique")
    answerable_values: list[str] = []
    adversarial_values: list[str] = []
    categories: list[str] = []
    for row in rows:
        kind = row.get("kind")
        if kind == "answerable":
            query = row.get("query")
            if query is None and row.get("query_source") == "first_chunk_informative_terms":
                terms = tuple(
                    term for term in index.document_tokens[0] if index.idf.get(term, 0.0) > 0.0
                )
                query = " ".join(dict.fromkeys(terms))
            if not query:
                raise RuntimeError("answerable golden row lacks a query/relevance label")
            answerable_values.append(str(query))
        elif kind == "unanswerable" and row.get("query"):
            category = str(row.get("category", ""))
            if category not in _REQUIRED_ADVERSARIAL_CATEGORIES:
                raise RuntimeError("golden row has an invalid adversarial category")
            adversarial_values.append(str(row["query"]))
            categories.append(category)
        else:
            raise RuntimeError("golden row has invalid relevance label")
    if set(categories) != _REQUIRED_ADVERSARIAL_CATEGORIES:
        missing = sorted(_REQUIRED_ADVERSARIAL_CATEGORIES - set(categories))
        raise RuntimeError(f"golden evaluation categories are incomplete:{','.join(missing)}")
    answerable = tuple(answerable_values)
    adversarial = tuple(adversarial_values)
    source_sha256 = sha256_file(path)
    evaluation_hash = sha256_bytes(
        canonical_json(
            {
                "answerable": answerable,
                "adversarial": adversarial,
                "categories": categories,
                "source_sha256": source_sha256,
                "mode": index.mode,
                "models": index.model_hashes or {},
            }
        )
    )
    return (
        answerable,
        adversarial,
        source,
        source_sha256,
        evaluation_hash,
        tuple(categories),
    )


def load_runtime_models(
    settings: Settings, index: BM25Index
) -> tuple[DenseEncoder | None, Reranker | None]:
    if index.mode == "bm25_only":
        return None, None
    embedder_model = acquire_locked_model(settings.workspace, settings.models["embedder"])
    embedder_hash = sha256_bytes(
        canonical_json(
            {
                "repository": embedder_model.repository,
                "revision": embedder_model.revision,
                "files": embedder_model.files,
            }
        )
    )
    if (index.model_hashes or {}).get("embedder") != embedder_hash or (index.model_files or {}).get(
        "embedder"
    ) != embedder_model.files:
        raise RuntimeError("embedder files do not match index")
    embedder: DenseEncoder = BGEDenseEncoder(
        embedder_model, str(settings.models["embedder"]["query_prefix"])
    )
    reranker: Reranker | None = None
    if index.mode == "full":
        reranker_model = acquire_locked_model(settings.workspace, settings.models["reranker"])
        reranker_hash = sha256_bytes(
            canonical_json(
                {
                    "repository": reranker_model.repository,
                    "revision": reranker_model.revision,
                    "files": reranker_model.files,
                }
            )
        )
        if (index.model_hashes or {}).get("reranker") != reranker_hash or (
            index.model_files or {}
        ).get("reranker") != reranker_model.files:
            raise RuntimeError("reranker files do not match index")
        reranker = BGEReranker(reranker_model)
    return embedder, reranker


def calibrate(
    settings: Settings,
    index: BM25Index | None = None,
    *,
    golden: Path | None = None,
    activate: bool = True,
    embedder: DenseEncoder | None = None,
    reranker: Reranker | None = None,
) -> Calibration:
    index = index or load_index(settings)
    (
        answerable_queries,
        adversarial_queries,
        evaluation_source,
        evaluation_source_sha256,
        evaluation_hash,
        adversarial_categories,
    ) = _queries(settings, index, golden)
    if not answerable_queries or not adversarial_queries:
        raise RuntimeError("calibration requires answerable and adversarial rows")
    if index.mode != "bm25_only" and embedder is None:
        embedder, loaded_reranker = load_runtime_models(settings, index)
        reranker = reranker or loaded_reranker
    lexical_index = replace(index, mode="bm25_only")
    raw_answerable = [retrieve(lexical_index, query, top_k=1)[0] for query in answerable_queries]
    viable_pairs = [
        (query, hit)
        for query, hit in zip(answerable_queries, raw_answerable, strict=True)
        if hit.informative_overlap >= 2 and hit.bm25_score > 0
    ]
    # A fixture or staged workspace may not contain optional supplemental sources
    # represented in the shared golden file. Calibrate only rows that have a real
    # lexical candidate; strict preparation still activates every configured source.
    if not viable_pairs:
        raise RuntimeError("calibration has no viable answerable rows")
    answerable_queries = tuple(query for query, _ in viable_pairs)
    viable = [hit for _, hit in viable_pairs]
    raw_adversarial = [retrieve(lexical_index, query, top_k=1)[0] for query in adversarial_queries]
    minimum_true_score = min(hit.bm25_score for hit in viable)
    coverage_floor = min(hit.lexical_coverage for hit in viable) * 0.8
    # A low-coverage adversarial hit cannot pass the lexical coverage gate, so it
    # should not raise the raw-score floor and suppress valid short questions.
    maximum_false_score = max(
        (
            hit.bm25_score
            for hit in raw_adversarial
            if hit.lexical_coverage >= coverage_floor and hit.informative_overlap >= 1
        ),
        default=0.0,
    )
    raw_floor = max(maximum_false_score + 1e-9, minimum_true_score * 0.5)
    candidates = [SupportThresholds(coverage_floor, raw_floor, 2)]
    if index.mode != "bm25_only":
        dense_grid = [float(item) for item in settings.rag["support"]["dense_grid"]]
        rerank_grid = [float(item) for item in settings.rag["support"]["rerank_grid"]]
        if index.mode == "mvp_hybrid":
            candidates = [
                SupportThresholds(coverage_floor, raw_floor, 2, dense) for dense in dense_grid
            ]
        else:
            candidates = [
                SupportThresholds(coverage_floor, raw_floor, 2, dense, rerank_value)
                for dense in dense_grid
                for rerank_value in rerank_grid
            ]
    scored = []
    for thresholds in candidates:
        answers = sum(
            retrieve(index, query, thresholds, top_k=1, embedder=embedder, reranker=reranker)[
                0
            ].supported
            for query in answerable_queries
        )
        false = sum(
            retrieve(index, query, thresholds, top_k=1, embedder=embedder, reranker=reranker)[
                0
            ].supported
            for query in adversarial_queries
        )
        if false == 0:
            scored.append((answers, thresholds))
    if not scored:
        raise RuntimeError("calibration found no zero-false-support threshold")
    answerable_supported, thresholds = max(
        scored,
        key=lambda item: (item[0], item[1].dense_score or -1.0, item[1].rerank_score or -1.0),
    )
    result = Calibration(
        index.fingerprint,
        index.corpus_fingerprint,
        settings.config_hash("rag", "models"),
        evaluation_source,
        evaluation_source_sha256,
        evaluation_hash,
        thresholds,
        len(answerable_queries),
        answerable_supported,
        len(adversarial_queries),
        adversarial_categories,
        0,
        index.mode,
        index.model_hashes or {},
    )
    if not result.ready:
        raise RuntimeError("calibration did not satisfy zero-false-support readiness")
    if activate:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = settings.state_dir / ".bm25_calibration.json.tmp"
        temporary.write_bytes(canonical_json(result.as_dict()) + b"\n")
        os.replace(temporary, settings.state_dir / "bm25_calibration.json")
    return result


def load_calibration(settings: Settings, index: BM25Index | None = None) -> Calibration:
    index = index or load_index(settings)
    path = settings.state_dir / "bm25_calibration.json"
    if not path.is_file():
        raise RuntimeError("BM25 calibration is missing")
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw = value["thresholds"]
    thresholds = SupportThresholds(
        float(raw["lexical_coverage"]),
        float(raw["raw_score"]),
        int(raw["minimum_overlap"]),
        float(raw["dense_score"]) if raw.get("dense_score") is not None else None,
        float(raw["rerank_score"]) if raw.get("rerank_score") is not None else None,
    )
    result = Calibration(
        str(value["index_fingerprint"]),
        str(value["corpus_fingerprint"]),
        str(value["rag_config_hash"]),
        str(value["evaluation_source"]),
        str(value["evaluation_source_sha256"]),
        str(value["evaluation_hash"]),
        thresholds,
        int(value["answerable_count"]),
        int(value["answerable_supported"]),
        int(value["adversarial_count"]),
        tuple(str(item) for item in value["adversarial_categories"]),
        int(value["false_support"]),
        str(value.get("retrieval_mode", "bm25_only")),
        {str(key): str(item) for key, item in value.get("model_hashes", {}).items()},
    )
    source = Path(result.evaluation_source)
    if not source.is_absolute():
        source = settings.workspace / source
    (
        _,
        _,
        expected_source,
        expected_source_sha256,
        expected_evaluation,
        expected_categories,
    ) = _queries(settings, index, source)
    expected = (
        result.index_fingerprint == index.fingerprint
        and result.corpus_fingerprint == index.corpus_fingerprint
        and result.rag_config_hash == settings.config_hash("rag", "models")
        and result.retrieval_mode == index.mode
        and result.model_hashes == (index.model_hashes or {})
        and result.evaluation_source == expected_source
        and result.evaluation_source_sha256 == expected_source_sha256
        and result.evaluation_hash == expected_evaluation
        and result.adversarial_categories == expected_categories
        and result.ready
    )
    if not expected:
        raise RuntimeError("BM25 calibration is stale or invalid")
    return result


def is_ready(settings: Settings) -> bool:
    try:
        load_calibration(settings)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True


@dataclass(frozen=True)
class SupportCalibration:
    model_hash: str
    evaluation_hash: str
    construction_hash: str
    config_hash: str
    threshold: float
    supported_count: int
    supported_accepted: int
    unsupported_count: int
    false_accepts: int

    @property
    def ready(self) -> bool:
        return self.supported_accepted > 0 and self.false_accepts == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_hash": self.model_hash,
            "evaluation_hash": self.evaluation_hash,
            "construction_hash": self.construction_hash,
            "config_hash": self.config_hash,
            "threshold": self.threshold,
            "supported_count": self.supported_count,
            "supported_accepted": self.supported_accepted,
            "unsupported_count": self.unsupported_count,
            "false_accepts": self.false_accepts,
            "ready": self.ready,
        }


def _support_inputs(settings: Settings) -> tuple[Path, str, str, str, list[dict[str, Any]]]:
    evaluation = settings.workspace / "eval" / "support_verifier.jsonl"
    rows = [
        json.loads(line)
        for line in evaluation.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({str(row.get("id", "")) for row in rows}) != len(rows):
        raise RuntimeError("support evaluation IDs must be nonempty and unique")
    if any(row.get("label") not in {"supported", "unsupported"} for row in rows):
        raise RuntimeError("support evaluation has an invalid label")
    lock = settings.models["support_verifier"]
    model_hash = sha256_bytes(
        canonical_json(
            {
                "repository": lock["repository"],
                "revision": lock["revision"],
                "files": lock["files"],
            }
        )
    )
    construction_hash = sha256_bytes(
        canonical_json(
            {
                "version": settings.rag["support"]["claim_calibration_version"],
                "maximum_tokens": lock["maximum_tokens"],
                "premise_stride": lock["premise_stride"],
                "entailment_label": lock["entailment_label"],
                "claim_scope": "cited-context-only-v1",
            }
        )
    )
    config_hash = sha256_bytes(
        canonical_json(
            {
                "grid": settings.rag["support"]["claim_threshold_grid"],
                "model_hash": model_hash,
                "construction_hash": construction_hash,
            }
        )
    )
    return evaluation, model_hash, construction_hash, config_hash, rows


def calibrate_support(
    settings: Settings, scorer: Any | None = None, *, activate: bool = True
) -> SupportCalibration:
    from acharya.rag.grounding import TransformersClaimScorer

    evaluation, model_hash, construction_hash, config_hash, rows = _support_inputs(settings)
    scorer = scorer or TransformersClaimScorer(
        settings.workspace, settings.models["support_verifier"]
    )
    values = [
        (
            str(row["label"]),
            float(scorer.score((str(row["premise"]),), str(row["hypothesis"]))),
        )
        for row in rows
    ]
    supported = tuple(score for label, score in values if label == "supported")
    unsupported = tuple(score for label, score in values if label == "unsupported")
    if not supported or not unsupported:
        raise RuntimeError("support calibration requires both labels")
    grid = settings.rag["support"]["claim_threshold_grid"]
    minimum = round(float(grid["minimum"]) * 100)
    maximum = round(float(grid["maximum"]) * 100)
    step = round(float(grid["step"]) * 100)
    candidates: list[tuple[int, float]] = []
    for value in range(minimum, maximum + 1, step):
        threshold = value / 100.0
        if sum(score >= threshold for score in unsupported) == 0:
            candidates.append((sum(score >= threshold for score in supported), threshold))
    if not candidates:
        raise RuntimeError("support calibration found no zero-false-accept threshold")
    supported_accepted, threshold = max(candidates, key=lambda item: (item[0], item[1]))
    result = SupportCalibration(
        model_hash=model_hash,
        evaluation_hash=sha256_file(evaluation),
        construction_hash=construction_hash,
        config_hash=config_hash,
        threshold=threshold,
        supported_count=len(supported),
        supported_accepted=supported_accepted,
        unsupported_count=len(unsupported),
        false_accepts=0,
    )
    if not result.ready:
        raise RuntimeError("support calibration did not retain any supported claims")
    if activate:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = settings.state_dir / ".support_calibration.json.tmp"
        temporary.write_bytes(canonical_json(result.as_dict()) + b"\n")
        os.replace(temporary, settings.state_dir / "support_calibration.json")
    return result


def load_support_calibration(settings: Settings) -> SupportCalibration:
    evaluation, model_hash, construction_hash, config_hash, _ = _support_inputs(settings)
    path = settings.state_dir / "support_calibration.json"
    if not path.is_file():
        raise RuntimeError("support calibration is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    result = SupportCalibration(
        model_hash=str(value["model_hash"]),
        evaluation_hash=str(value["evaluation_hash"]),
        construction_hash=str(value["construction_hash"]),
        config_hash=str(value["config_hash"]),
        threshold=float(value["threshold"]),
        supported_count=int(value["supported_count"]),
        supported_accepted=int(value["supported_accepted"]),
        unsupported_count=int(value["unsupported_count"]),
        false_accepts=int(value["false_accepts"]),
    )
    if not (
        result.ready
        and 0.5 <= result.threshold <= 0.99
        and result.model_hash == model_hash
        and result.evaluation_hash == sha256_file(evaluation)
        and result.construction_hash == construction_hash
        and result.config_hash == config_hash
    ):
        raise RuntimeError("support calibration is stale or invalid")
    return result


def support_is_ready(settings: Settings) -> bool:
    try:
        load_support_calibration(settings)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True
