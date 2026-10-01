from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from acharya.config import Settings
from acharya.ingest.pipeline import build_corpus
from acharya.rag.evaluate import calibrate
from acharya.rag.index import build_index


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace_factory(tmp_path: Path, project_root: Path) -> Callable[[], Path]:
    def create() -> Path:
        root = tmp_path / "workspace"
        root.mkdir(exist_ok=True)
        shutil.copy2(project_root / "pyproject.toml", root / "pyproject.toml")
        shutil.copytree(project_root / "configs", root / "configs")
        shutil.copytree(project_root / "eval", root / "eval")
        (root / "data").mkdir()
        shutil.copy2(project_root / "data" / "dataset V1.0.pdf", root / "data" / "dataset V1.0.pdf")
        return root

    return create


@pytest.fixture
def built_workspace(workspace_factory: Callable[[], Path]) -> Path:
    root = workspace_factory()
    settings = Settings.load(root)
    build_corpus(settings)
    index = build_index(settings)
    calibrate(settings, index)
    return root
