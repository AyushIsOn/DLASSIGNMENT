from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.rag.index import build_index, load_index, tokenize


def test_index_is_deterministic_and_raw(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    first = build_index(settings)
    second = build_index(settings)
    assert first.fingerprint == second.fingerprint
    assert tuple(item.chunk_id for item in first.documents) == tuple(
        sorted(item.chunk_id for item in first.documents)
    )
    assert load_index(settings).fingerprint == first.fingerprint
    assert first.score(tokenize("Eka kusta dosha"), 0) != 0.75
