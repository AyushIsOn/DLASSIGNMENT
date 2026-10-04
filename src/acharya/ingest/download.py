"""Pinned Kaggle acquisition with secure extraction and strict cache validation."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from acharya.config import ConfigurationError, sha256_file

PROHIBITED_DATASET = "rcratos/ayurveda-texts-english/1"


class KaggleClient(Protocol):
    def dataset_status(self, dataset: str, format: str | None = None) -> str: ...
    def dataset_list_files(
        self, dataset: str, page_token: str | None = None, page_size: int = 20
    ) -> object: ...
    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
        licenses: list[str] | None = None,
    ) -> None: ...


class AcquisitionError(RuntimeError):
    def __init__(self, dataset_id: str, category: str, cache_decision: str) -> None:
        self.dataset_id = dataset_id
        self.category = category
        self.cache_decision = cache_decision
        super().__init__(f"dataset acquisition blocked: {dataset_id}:{category}:{cache_decision}")


@dataclass(frozen=True)
class Acquisition:
    dataset_id: str
    archive: Path
    extracted: Path
    archive_sha256: str
    member_sha256: dict[str, str]
    cache_used: bool
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "archive": str(self.archive),
            "extracted": str(self.extracted),
            "archive_sha256": self.archive_sha256,
            "member_sha256": self.member_sha256,
            "cache_used": self.cache_used,
            "metadata": self.metadata,
        }


def _failure_category(error: BaseException) -> str:
    if isinstance(error, SystemExit):
        return "authentication"
    text = f"{type(error).__name__} {error}".casefold()
    if any(item in text for item in ("401", "403", "auth", "credential", "token")):
        return "authentication"
    if "429" in text or "rate" in text:
        return "rate_limit"
    if any(item in text for item in ("dns", "name resolution", "connection")):
        return "network"
    if any(item in text for item in ("tls", "ssl", "certificate")):
        return "tls"
    if "timeout" in text:
        return "timeout"
    if any(f"{code}" in text for code in range(500, 600)):
        return "server_error"
    return "external_error"


def _files_snapshot(value: object) -> list[dict[str, object]]:
    files = getattr(value, "files", None) or getattr(value, "dataset_files", None) or []
    result: list[dict[str, object]] = []
    for item in files:
        name = getattr(item, "name", None) or getattr(item, "ref", None) or ""
        size = getattr(item, "total_bytes", None) or getattr(item, "size", None)
        result.append({"name": str(name), "size": int(size) if size is not None else None})
    return sorted(result, key=lambda item: str(item["name"]))


def _snapshot(client: KaggleClient, dataset_id: str) -> dict[str, object]:
    status = json.loads(client.dataset_status(dataset_id, format="json(current_version_number)"))
    listing = client.dataset_list_files(dataset_id, page_size=100)
    return {
        "requested_version": int(dataset_id.rsplit("/", 1)[1]),
        "current_version": int(status["current_version_number"]),
        "files": _files_snapshot(listing),
    }


def _cache_valid(archive: Path, spec: dict[str, Any]) -> bool:
    return archive.is_file() and sha256_file(archive) == str(spec["archive_sha256"])


def _validate_and_extract(
    archive: Path, destination: Path, spec: dict[str, Any], limits: dict[str, Any]
) -> dict[str, str]:
    if archive.stat().st_size > int(limits["maximum_archive_bytes"]):
        raise ConfigurationError("archive_size_limit")
    expected = {str(name): str(digest) for name, digest in spec["members"].items()}
    hashes: dict[str, str] = {}
    expanded = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > int(limits["maximum_members"]):
            raise ConfigurationError("archive_member_limit")
        by_name: dict[str, zipfile.ZipInfo] = {}
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ConfigurationError("unsafe_archive_path")
            file_type = stat.S_IFMT(mode)
            if member.is_dir() or stat.S_ISLNK(mode) or (file_type and not stat.S_ISREG(mode)):
                raise ConfigurationError("unsafe_archive_member")
            if path.suffix.casefold() not in {
                str(item).casefold() for item in limits["allowed_extensions"]
            }:
                raise ConfigurationError("unexpected_archive_extension")
            if member.file_size > int(limits["maximum_member_bytes"]):
                raise ConfigurationError("archive_member_size_limit")
            expanded += member.file_size
            by_name[member.filename] = member
        if expanded > int(limits["maximum_expanded_bytes"]):
            raise ConfigurationError("archive_expanded_size_limit")
        if set(by_name) != set(expected):
            raise ConfigurationError("archive_member_set_mismatch")
        destination.mkdir(parents=True, exist_ok=False)
        for name, digest in sorted(expected.items()):
            target = destination / name
            with bundle.open(by_name[name]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            actual = sha256_file(target)
            if actual != digest:
                raise ConfigurationError(f"member_hash_mismatch:{name}")
            hashes[name] = actual
    return hashes


def _atomic_replace_directory(source: Path, target: Path) -> None:
    """Replace a directory while retaining and rolling back the previous target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup-", dir=target.parent))
        backup.rmdir()
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except BaseException:
        if backup is not None and backup.exists():
            os.replace(backup, target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def acquire_dataset(
    workspace: Path,
    dataset_id: str,
    spec: dict[str, Any],
    limits: dict[str, Any],
    *,
    client: KaggleClient | None = None,
    strict: bool = True,
    prefer_cache: bool = True,
) -> Acquisition:
    if (
        dataset_id == PROHIBITED_DATASET
        or dataset_id.rsplit("/", 1)[0] == PROHIBITED_DATASET.rsplit("/", 1)[0]
    ):
        raise ConfigurationError("prohibited_dataset")
    expected_id = f"{spec['owner']}/{spec['slug']}/{int(spec['version'])}"
    if dataset_id != expected_id:
        raise ConfigurationError("dataset_version_mismatch")
    root = workspace / "data" / "raw" / f"{spec['owner']}__{spec['slug']}" / str(spec["version"])
    archive = root / f"{spec['slug']}.zip"
    extracted = root / "files"
    if prefer_cache and _cache_valid(archive, spec) and extracted.is_dir():
        hashes = {str(name): sha256_file(extracted / str(name)) for name in spec["members"]}
        if hashes == {str(name): str(value) for name, value in spec["members"].items()}:
            return Acquisition(
                dataset_id,
                archive,
                extracted,
                sha256_file(archive),
                hashes,
                True,
                {"source": "validated_cache", "requested_version": int(spec["version"])},
            )
    if client is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            client = api
        except BaseException as error:
            category = _failure_category(error)
            retryable = {
                "authentication",
                "rate_limit",
                "network",
                "tls",
                "timeout",
                "server_error",
            }
            if strict and category in retryable and _cache_valid(archive, spec):
                return _activate_cached(workspace, dataset_id, spec, limits, archive)
            raise AcquisitionError(dataset_id, category, "cache_missing_or_invalid") from None
    staging_parent = workspace / "data" / "raw" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="kaggle-", dir=staging_parent))
    try:
        before = _snapshot(client, dataset_id)
        client.dataset_download_files(
            dataset_id, path=str(staging), force=True, quiet=True, unzip=False
        )
        after = _snapshot(client, dataset_id)
        if before != after:
            raise ConfigurationError("dataset_metadata_changed_during_download")
        downloaded = staging / f"{spec['slug']}.zip"
        if not downloaded.is_file() or sha256_file(downloaded) != str(spec["archive_sha256"]):
            raise ConfigurationError("archive_hash_mismatch")
        files = staging / "files"
        hashes = _validate_and_extract(downloaded, files, spec, limits)
        root.parent.mkdir(parents=True, exist_ok=True)
        _atomic_replace_directory(staging, root)
        return Acquisition(
            dataset_id,
            root / downloaded.name,
            root / "files",
            str(spec["archive_sha256"]),
            hashes,
            False,
            before,
        )
    except ConfigurationError:
        raise
    except BaseException as error:
        category = _failure_category(error)
        retryable = {
            "authentication",
            "rate_limit",
            "network",
            "tls",
            "timeout",
            "server_error",
        }
        if strict and category in retryable and _cache_valid(archive, spec):
            return _activate_cached(workspace, dataset_id, spec, limits, archive)
        raise AcquisitionError(dataset_id, category, "cache_missing_or_invalid") from None
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _activate_cached(
    workspace: Path, dataset_id: str, spec: dict[str, Any], limits: dict[str, Any], archive: Path
) -> Acquisition:
    staging_parent = workspace / "data" / "raw" / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="cache-", dir=staging_parent))
    try:
        hashes = _validate_and_extract(archive, staging / "files", spec, limits)
        extracted = archive.parent / "files"
        _atomic_replace_directory(staging / "files", extracted)
        return Acquisition(
            dataset_id,
            archive,
            extracted,
            sha256_file(archive),
            hashes,
            True,
            {"source": "validated_cache", "requested_version": int(spec["version"])},
        )
    except BaseException as error:
        raise AcquisitionError(dataset_id, "cache_validation", "cache_invalid") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def blocker_record(error: AcquisitionError) -> dict[str, object]:
    return {
        "dataset_id": error.dataset_id,
        "attempted_version": int(error.dataset_id.rsplit("/", 1)[1]),
        "failure_category": error.category,
        "cache_decision": error.cache_decision,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "redacted": True,
    }
