from __future__ import annotations

import pytest

from acharya.config import Settings, canonical_json
from acharya.ingest.pipeline import active_corpus, build_corpus


def test_real_build_is_deterministic_and_atomic(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    first = build_corpus(settings)
    second = build_corpus(settings)
    assert first.fingerprint == second.fingerprint
    assert first.chunk_ids == second.chunk_ids
    assert first.duplicate_components == second.duplicate_components
    pointer = canonical_json(active_corpus(settings))
    with pytest.raises(RuntimeError, match="injected"):
        build_corpus(settings, inject_failure=True)
    assert canonical_json(active_corpus(settings)) == pointer
    assert first.eligible_count > 0
