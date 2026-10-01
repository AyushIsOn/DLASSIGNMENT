from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from acharya.config import ConfigurationError, sha256_file
from acharya.ingest import download
from acharya.ingest.download import AcquisitionError, acquire_dataset


class FakeClient:
    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.calls: list[tuple[str, str]] = []

    def dataset_status(self, dataset: str, format: str | None = None) -> str:
        self.calls.append(("status", dataset))
        return json.dumps({"current_version_number": 1})

    def dataset_list_files(
        self, dataset: str, page_token: str | None = None, page_size: int = 20
    ) -> object:
        self.calls.append(("list", dataset))
        return SimpleNamespace(files=[SimpleNamespace(name="records.csv", total_bytes=8)])

    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
        licenses: list[str] | None = None,
    ) -> None:
        self.calls.append(("download", dataset))
        assert path is not None and unzip is False
        Path(path, "sample.zip").write_bytes(self.archive.read_bytes())


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("records.csv", "a,b\n1,2\n")
    member_hash = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    spec: dict[str, object] = {
        "owner": "owner",
        "slug": "sample",
        "version": 1,
        "archive_sha256": sha256_file(archive),
        "members": {"records.csv": member_hash},
    }
    limits: dict[str, object] = {
        "maximum_archive_bytes": 10000,
        "maximum_expanded_bytes": 10000,
        "maximum_member_bytes": 10000,
        "maximum_members": 2,
        "allowed_extensions": [".csv"],
    }
    return archive, spec, limits


def test_versioned_download_before_and_after_and_hashes(tmp_path: Path) -> None:
    archive, spec, limits = _fixture(tmp_path)
    client = FakeClient(archive)
    result = acquire_dataset(tmp_path / "workspace", "owner/sample/1", spec, limits, client=client)
    assert client.calls == [
        ("status", "owner/sample/1"),
        ("list", "owner/sample/1"),
        ("download", "owner/sample/1"),
        ("status", "owner/sample/1"),
        ("list", "owner/sample/1"),
    ]
    assert result.member_sha256 == spec["members"]


def test_prohibited_dataset_is_rejected_before_client_use(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="prohibited"):
        acquire_dataset(tmp_path, "rcratos/ayurveda-texts-english/1", {}, {})


def test_download_activation_failure_restores_previous_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, spec, limits = _fixture(tmp_path)
    workspace = tmp_path / "workspace"
    target = workspace / "data" / "raw" / "owner__sample" / "1"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_text("previous-valid-cache", encoding="utf-8")
    original_replace = download.os.replace

    def fail_new_activation(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == target and source_path.name.startswith("kaggle-"):
            raise OSError("injected activation failure")
        original_replace(source, destination)

    monkeypatch.setattr(download.os, "replace", fail_new_activation)
    with pytest.raises(AcquisitionError):
        acquire_dataset(
            workspace,
            "owner/sample/1",
            spec,
            limits,
            client=FakeClient(archive),
            prefer_cache=False,
        )
    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "previous-valid-cache"
