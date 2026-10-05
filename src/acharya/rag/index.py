"""Atomic raw-BM25, dense, and hybrid retrieval indexes."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from acharya.config import Settings, canonical_json, sha256_bytes, sha256_file
from acharya.ingest.pipeline import active_corpus
from acharya.rag.embed import BGEDenseEncoder, DenseEncoder, acquire_locked_model

RetrievalMode = Literal["bm25_only", "mvp_hybrid", "full"]
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "which",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token.casefold() for token in _TOKEN.findall(text) if token.casefold() not in STOPWORDS
    )


@dataclass(frozen=True)
class BM25Document:
    chunk_id: str
    source: str
    page: int
    question: str
    text: str
    role: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BM25Document:
        return cls(
            str(value["chunk_id"]),
            str(value["source"]),
            int(value["page"]),
            str(value["question"]),
            str(value["text"]),
            str(value["role"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "question": self.question,
            "text": self.text,
            "role": self.role,
        }


@dataclass(frozen=True)
class BM25Index:
    fingerprint: str
    corpus_fingerprint: str
    config_hash: str
    documents: tuple[BM25Document, ...]
    document_tokens: tuple[tuple[str, ...], ...]
    idf: dict[str, float]
    average_length: float
    k1: float
    b: float
    mode: RetrievalMode = "bm25_only"
    dense_vectors: tuple[tuple[float, ...], ...] = ()
    model_hashes: dict[str, str] | None = None
    model_files: dict[str, dict[str, str]] | None = None

    def score(self, query_tokens: tuple[str, ...], document_index: int) -> float:
        frequencies = Counter(self.document_tokens[document_index])
        length = len(self.document_tokens[document_index])
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / max(self.average_length, 1.0)
            )
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "fingerprint": self.fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "config_hash": self.config_hash,
            "mode": self.mode,
            "documents": [item.as_dict() for item in self.documents],
            "document_tokens": [list(item) for item in self.document_tokens],
            "idf": self.idf,
            "average_length": self.average_length,
            "k1": self.k1,
            "b": self.b,
            "dense_vectors": [list(item) for item in self.dense_vectors],
            "model_hashes": self.model_hashes or {},
            "model_files": self.model_files or {},
        }


def _payload(
    corpus_fingerprint: str,
    config_hash: str,
    mode: RetrievalMode,
    documents: tuple[BM25Document, ...],
    document_tokens: tuple[tuple[str, ...], ...],
    idf: dict[str, float],
    average_length: float,
    k1: float,
    b: float,
    dense_vectors: tuple[tuple[float, ...], ...],
    model_hashes: dict[str, str],
    model_files: dict[str, dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "corpus_fingerprint": corpus_fingerprint,
        "config_hash": config_hash,
        "mode": mode,
        "documents": [item.as_dict() for item in documents],
        "document_tokens": [list(item) for item in document_tokens],
        "idf": idf,
        "average_length": average_length,
        "k1": k1,
        "b": b,
        "dense_vectors": [list(item) for item in dense_vectors],
        "model_hashes": model_hashes,
        "model_files": model_files,
    }


def retrieval_passage(document: BM25Document) -> str:
    """Use existing source question metadata, never evaluation reference answers."""
    if not document.question.strip():
        return document.text
    return f"Question: {document.question}\nPassage: {document.text}"


def build_index(
    settings: Settings,
    retrieval_mode: RetrievalMode = "bm25_only",
    *,
    embedder: DenseEncoder | None = None,
    reranker: object | None = None,
    allow_bm25_fallback: bool = False,
    inject_failure: bool = False,
) -> BM25Index:
    if retrieval_mode not in {"bm25_only", "mvp_hybrid", "full"}:
        raise ValueError(f"unsupported retrieval mode:{retrieval_mode}")
    active = active_corpus(settings)
    if active is None:
        raise RuntimeError("no active corpus; run corpus build first")
    corpus_path = Path(str(active["path"]))
    if not corpus_path.is_absolute():
        corpus_path = settings.workspace / corpus_path
    corpus = json.loads((corpus_path / "corpus.json").read_text(encoding="utf-8"))
    documents = tuple(
        sorted(
            (BM25Document.from_dict(item) for item in corpus["chunks"]),
            key=lambda item: item.chunk_id,
        )
    )
    if not documents:
        raise RuntimeError("active corpus has no eligible documents")
    document_tokens = tuple(tokenize(f"{item.question} {item.text}") for item in documents)
    count = len(documents)
    frequencies = Counter(term for tokens in document_tokens for term in set(tokens))
    idf = {
        term: math.log((count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in sorted(frequencies.items())
    }
    average_length = sum(len(item) for item in document_tokens) / count
    bm25 = settings.rag["bm25"]
    k1, b = float(bm25["k1"]), float(bm25["b"])
    dense_vectors: tuple[tuple[float, ...], ...] = ()
    model_hashes: dict[str, str] = {}
    model_files: dict[str, dict[str, str]] = {}
    try:
        if retrieval_mode != "bm25_only":
            if embedder is None:
                resolved = acquire_locked_model(settings.workspace, settings.models["embedder"])
                model_hashes["embedder"] = sha256_bytes(
                    canonical_json(
                        {
                            "repository": resolved.repository,
                            "revision": resolved.revision,
                            "files": resolved.files,
                        }
                    )
                )
                model_files["embedder"] = resolved.files
                embedder = BGEDenseEncoder(
                    resolved, str(settings.models["embedder"]["query_prefix"])
                )
            else:
                model_hashes["embedder"] = "injected-test-double"
                model_files["embedder"] = {"injected": "injected-test-double"}
            model_hashes["passage_format"] = sha256_bytes(b"source-question-and-text-v1")
            vectors = embedder.encode_passages([retrieval_passage(item) for item in documents])
            if len(vectors) != count or any(not vector for vector in vectors):
                raise RuntimeError("dense embedding count mismatch")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise RuntimeError("dense embedding dimension mismatch")
            dense_vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
            if retrieval_mode == "full":
                if reranker is None:
                    resolved_reranker = acquire_locked_model(
                        settings.workspace, settings.models["reranker"]
                    )
                    model_hashes["reranker"] = sha256_bytes(
                        canonical_json(
                            {
                                "repository": resolved_reranker.repository,
                                "revision": resolved_reranker.revision,
                                "files": resolved_reranker.files,
                            }
                        )
                    )
                    model_files["reranker"] = resolved_reranker.files
                else:
                    model_hashes["reranker"] = "injected-test-double"
                    model_files["reranker"] = {"injected": "injected-test-double"}
    except BaseException:
        if allow_bm25_fallback and retrieval_mode == "mvp_hybrid":
            return build_index(settings, "bm25_only", inject_failure=inject_failure)
        raise
    config_hash = settings.config_hash("rag", "models")
    payload = _payload(
        str(corpus["fingerprint"]),
        config_hash,
        retrieval_mode,
        documents,
        document_tokens,
        idf,
        average_length,
        k1,
        b,
        dense_vectors,
        model_hashes,
        model_files,
    )
    fingerprint = sha256_bytes(canonical_json(payload))
    index = BM25Index(
        fingerprint,
        str(corpus["fingerprint"]),
        config_hash,
        documents,
        document_tokens,
        idf,
        average_length,
        k1,
        b,
        retrieval_mode,
        dense_vectors,
        model_hashes,
        model_files,
    )
    root = settings.workspace / "artifacts" / "indexes"
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".index-", dir=root))
    try:
        (staging / "index.json").write_bytes(canonical_json(index.as_dict()) + b"\n")
        (staging / "manifest.json").write_bytes(
            canonical_json(
                {
                    "fingerprint": fingerprint,
                    "index_sha256": sha256_file(staging / "index.json"),
                    "document_count": count,
                    "mode": retrieval_mode,
                    "model_hashes": model_hashes,
                    "model_files": model_files,
                }
            )
            + b"\n"
        )
        if inject_failure:
            raise RuntimeError("injected staged index failure")
        target = root / fingerprint
        if target.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, target)
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        pointer = settings.state_dir / ".active_index.json.tmp"
        pointer.write_bytes(
            canonical_json(
                {"fingerprint": fingerprint, "path": str(target), "mode": retrieval_mode}
            )
            + b"\n"
        )
        os.replace(pointer, settings.state_dir / "active_index.json")
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return index


def load_index(settings: Settings, expected_mode: RetrievalMode | None = None) -> BM25Index:
    pointer_path = settings.state_dir / "active_index.json"
    if not pointer_path.is_file():
        raise RuntimeError("no active index")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    value = json.loads((Path(pointer["path"]) / "index.json").read_text(encoding="utf-8"))
    fingerprint = str(value.pop("fingerprint"))
    documents = tuple(BM25Document.from_dict(item) for item in value["documents"])
    payload = {**value, "documents": [item.as_dict() for item in documents]}
    if sha256_bytes(canonical_json(payload)) != fingerprint:
        raise RuntimeError("index fingerprint mismatch")
    if value["config_hash"] != settings.config_hash("rag", "models"):
        raise RuntimeError("index configuration is stale")
    active = active_corpus(settings)
    if active is None or value["corpus_fingerprint"] != active["fingerprint"]:
        raise RuntimeError("index corpus is stale")
    mode: RetrievalMode = value["mode"]
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"active index mode mismatch:{mode}")
    vectors = tuple(tuple(float(item) for item in row) for row in value["dense_vectors"])
    if mode != "bm25_only" and len(vectors) != len(documents):
        raise RuntimeError("dense index count mismatch")
    return BM25Index(
        fingerprint,
        str(value["corpus_fingerprint"]),
        str(value["config_hash"]),
        documents,
        tuple(tuple(tokens) for tokens in value["document_tokens"]),
        {str(key): float(score) for key, score in value["idf"].items()},
        float(value["average_length"]),
        float(value["k1"]),
        float(value["b"]),
        mode,
        vectors,
        {str(key): str(item) for key, item in value["model_hashes"].items()},
        {
            str(role): {str(name): str(digest) for name, digest in files.items()}
            for role, files in value["model_files"].items()
        },
    )
