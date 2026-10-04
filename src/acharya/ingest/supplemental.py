"""Bounded, hash-pinned acquisition of public supplemental data."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from acharya.config import ConfigurationError, Settings, sha256_file
from acharya.ingest.download import _atomic_replace_directory, _validate_and_extract


def download_pinned_file(url: str, target: Path, expected: str, maximum: int) -> None:
    if target.is_file() and sha256_file(target) == expected:
        return
    if not url.startswith("https://"):
        raise ConfigurationError("source requires an HTTPS download URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AcharyaGPT-data/1.0"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
            request, timeout=60
        ) as response:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > maximum:
                    raise ConfigurationError("source download exceeds byte limit")
                output.write(block)
        if sha256_file(Path(temporary)) != expected:
            raise ConfigurationError("downloaded source hash mismatch")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def acquire_supplemental(settings: Settings) -> None:
    limits: dict[str, Any] = settings.datasets["kaggle"]["limits"]
    for spec in settings.datasets.get("supplemental", {}).values():
        archive = settings.workspace / str(spec["archive_relative_path"])
        url = ("https://www.kaggle.com/api/v1/datasets/download/"
               f"{spec['owner']}/{spec['slug']}?datasetVersionNumber={int(spec['version'])}")
        download_pinned_file(url, archive, str(spec["archive_sha256"]),
                             int(limits["maximum_archive_bytes"]))
        target = (settings.workspace / str(spec["relative_path"])).parent
        if all((target / name).is_file() and sha256_file(target / name) == digest
               for name, digest in spec["members"].items()):
            continue
        stage = Path(tempfile.mkdtemp(prefix=".extract-", dir=archive.parent))
        try:
            _validate_and_extract(archive, stage / "files", spec, limits)
            _atomic_replace_directory(stage / "files", target)
        finally:
            shutil.rmtree(stage)
    for dataset_id, spec in settings.datasets.get("text_sources", {}).items():
        target = settings.workspace / str(spec["relative_path"])
        if target.is_file() and sha256_file(target) == spec["sha256"]:
            continue
        if not spec.get("download_url"):
            raise ConfigurationError(f"tracked curated source missing or modified:{dataset_id}")
        download_pinned_file(str(spec["download_url"]), target, str(spec["sha256"]),
                             int(limits["maximum_member_bytes"]))
