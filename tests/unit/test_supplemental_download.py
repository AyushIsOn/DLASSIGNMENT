from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from acharya.config import ConfigurationError, Settings, sha256_bytes
from acharya.ingest.supplemental import acquire_supplemental, download_pinned_file


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


def test_cold_archive_extracts_all_members_and_repairs_cached_files(
    workspace_factory: Callable[[], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings.load(workspace_factory())
    members = {"source.csv": b"question,answer\nexample,answer\n", "source.xlsx": b"spreadsheet"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    payload = buffer.getvalue()
    settings.datasets["supplemental"] = {"fixture": {
        "owner": "fixture", "slug": "dataset", "version": 1,
        "archive_relative_path": "data/raw/fixture/archive.zip",
        "relative_path": "data/raw/fixture/files/source.csv",
        "archive_sha256": sha256_bytes(payload),
        "members": {name: sha256_bytes(content) for name, content in members.items()},
    }}
    settings.datasets["text_sources"] = {}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: io.BytesIO(payload))
    acquire_supplemental(settings)
    target = settings.workspace / "data/raw/fixture/files"
    assert {p.name: p.read_bytes() for p in target.iterdir()} == members
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("cache missed"))
    acquire_supplemental(settings)
    (target / "source.xlsx").write_bytes(b"damaged")
    acquire_supplemental(settings)
    assert (target / "source.xlsx").read_bytes() == members["source.xlsx"]
