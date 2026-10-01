"""Typed, secret-free configuration loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when immutable project configuration is invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration must be a mapping: {path.name}")
    return value


@dataclass(frozen=True)
class Settings:
    workspace: Path
    ingestion: dict[str, Any]
    rag: dict[str, Any]
    safety: dict[str, Any]
    datasets: dict[str, Any]
    models: dict[str, Any]

    @classmethod
    def load(cls, workspace: Path | str) -> Settings:
        root = Path(workspace).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise ConfigurationError(f"not an AcharyaGPT workspace: {root}")
        return cls(
            workspace=root,
            ingestion=load_yaml(root / "configs" / "ingestion.yaml"),
            rag=load_yaml(root / "configs" / "rag.yaml"),
            safety=load_yaml(root / "configs" / "safety.yaml"),
            datasets=load_yaml(root / "configs" / "datasets.yaml"),
            models=load_yaml(root / "configs" / "models.lock.yaml"),
        )

    def config_hash(self, *names: str) -> str:
        values = {name: getattr(self, name) for name in names}
        return sha256_bytes(canonical_json(values))

    @property
    def pdf_path(self) -> Path:
        return self.workspace / str(self.ingestion["pdf"]["relative_path"])

    @property
    def state_dir(self) -> Path:
        return self.workspace / "artifacts" / "state"
