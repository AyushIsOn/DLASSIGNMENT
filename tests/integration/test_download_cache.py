from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

from acharya.config import sha256_file
from acharya.ingest import download
from acharya.ingest.download import AcquisitionError, acquire_dataset


def test_only_hash_valid_cache_survives_network_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = workspace / "data" / "raw" / "owner__sample" / "1" / "sample.zip"
    cache.parent.mkdir(parents=True)
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("records.csv", "value\nvalid\n")
    shutil.copy2(source, cache)
    member = hashlib.sha256(b"value\nvalid\n").hexdigest()
    spec = {
        "owner": "owner",
        "slug": "sample",
        "version": 1,
        "archive_sha256": sha256_file(source),
        "members": {"records.csv": member},
    }
    limits = {
        "maximum_archive_bytes": 10000,
        "maximum_expanded_bytes": 10000,
        "maximum_member_bytes": 10000,
        "maximum_members": 2,
        "allowed_extensions": [".csv"],
    }
    result = acquire_dataset(workspace, "owner/sample/1", spec, limits, strict=True)
    assert result.cache_used
    cache.write_bytes(cache.read_bytes() + b"x")
    with pytest.raises(AcquisitionError, match="cache_missing_or_invalid"):
        acquire_dataset(workspace, "owner/sample/1", spec, limits, strict=True)


def test_cached_activation_failure_restores_previous_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    cache = workspace / "data" / "raw" / "owner__sample" / "1" / "sample.zip"
    cache.parent.mkdir(parents=True)
    with zipfile.ZipFile(cache, "w") as bundle:
        bundle.writestr("records.csv", "value\nvalid\n")
    old_files = cache.parent / "files"
    old_files.mkdir()
    (old_files / "sentinel.txt").write_text("previous-extraction", encoding="utf-8")
    spec = {
        "owner": "owner",
        "slug": "sample",
        "version": 1,
        "archive_sha256": sha256_file(cache),
        "members": {"records.csv": hashlib.sha256(b"value\nvalid\n").hexdigest()},
    }
    limits = {
        "maximum_archive_bytes": 10000,
        "maximum_expanded_bytes": 10000,
        "maximum_member_bytes": 10000,
        "maximum_members": 2,
        "allowed_extensions": [".csv"],
    }

    class FailingClient:
        def dataset_status(self, dataset: str, format: str | None = None) -> str:
            raise OSError("network connection failed")

    original_replace = download.os.replace

    def fail_new_activation(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == old_files and source_path.parent.name.startswith("cache-"):
            raise OSError("injected cache activation failure")
        original_replace(source, destination)

    monkeypatch.setattr(download.os, "replace", fail_new_activation)
    with pytest.raises(AcquisitionError, match="cache_invalid"):
        acquire_dataset(
            workspace,
            "owner/sample/1",
            spec,
            limits,
            client=FailingClient(),  # type: ignore[arg-type]
            strict=True,
            prefer_cache=False,
        )
    assert (old_files / "sentinel.txt").read_text(encoding="utf-8") == "previous-extraction"
