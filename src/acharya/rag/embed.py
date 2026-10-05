"""Locked BGE model acquisition, embedding, and reranking adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from acharya.config import sha256_file


class DenseEncoder(Protocol):
    def encode_passages(self, passages: list[str]) -> list[list[float]]: ...
    def encode_query(self, query: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class ResolvedModel:
    repository: str
    revision: str
    path: Path
    files: dict[str, str]


def acquire_locked_model(workspace: Path, lock: dict[str, Any]) -> ResolvedModel:
    if bool(lock.get("trust_remote_code", False)):
        raise RuntimeError("remote model code is prohibited")
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=str(lock["repository"]),
            revision=str(lock["revision"]),
            cache_dir=workspace / ".cache" / "huggingface",
        )
    )
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise RuntimeError(f"model snapshot is empty:{lock['repository']}")
    return ResolvedModel(str(lock["repository"]), str(lock["revision"]), path, files)


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class BGEDenseEncoder:
    def __init__(self, model: ResolvedModel, query_prefix: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(model.path), trust_remote_code=False)
        self._prefix = query_prefix

    def _encode(self, values: list[str]) -> list[list[float]]:
        encoded = self._model.encode(
            values, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [[float(item) for item in row] for row in encoded]

    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        return self._encode(passages)

    def encode_query(self, query: str) -> list[float]:
        return self._encode([f"{self._prefix}{query}"])[0]


class BGEReranker:
    def __init__(self, model: ResolvedModel) -> None:
        from sentence_transformers import CrossEncoder
        from torch.nn import Identity

        self._model = CrossEncoder(
            str(model.path), trust_remote_code=False, activation_fn=Identity()
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        values = self._model.predict(
            [(query, passage) for passage in passages], apply_softmax=False, show_progress_bar=False
        )
        result = []
        for value in values:
            logit = float(value)
            result.append(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))
        return result
