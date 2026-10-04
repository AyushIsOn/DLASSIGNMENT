from __future__ import annotations

import io
from pathlib import Path

import pytest

from acharya.config import ConfigurationError, sha256_bytes
from acharya.ingest.supplemental import download_pinned_file


def test_download_is_bounded_hashed_atomic_and_cacheable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "source.txt"
    target.write_bytes(b"previous")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: io.BytesIO(b"new data"))
    digest = sha256_bytes(b"new data")
    with pytest.raises(ConfigurationError, match="byte limit"):
        download_pinned_file("https://example.invalid/data", target, digest, 2)
    assert target.read_bytes() == b"previous"
    with pytest.raises(ConfigurationError, match="hash mismatch"):
        download_pinned_file("https://example.invalid/data", target, "bad", 100)
    assert target.read_bytes() == b"previous"
    download_pinned_file("https://example.invalid/data", target, digest, 100)
    assert target.read_bytes() == b"new data"
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("cache missed"))
    download_pinned_file("https://example.invalid/data", target, digest, 100)
    assert list(tmp_path.iterdir()) == [target]
