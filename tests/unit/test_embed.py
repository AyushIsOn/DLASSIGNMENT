from __future__ import annotations

import pytest

from acharya.rag.embed import cosine


def test_cosine_is_deterministic_and_rejects_mismatched_dimensions() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        cosine([1.0], [1.0, 2.0])
