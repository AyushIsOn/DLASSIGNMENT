from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from acharya import bundle
from acharya.config import canonical_json, sha256_file
from acharya.lightning import PreflightError


def _completed_preparation(root: Path) -> str:
    fingerprint = "a" * 64
    processed = root / "data" / "processed" / fingerprint
    reports = root / "artifacts" / "preparation" / fingerprint
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (processed / name).write_text('{"id":"one","messages":[]}\n', encoding="utf-8")
    (processed / "corpus.json").write_text(
        json.dumps({"fingerprint": fingerprint, "chunks": []}), encoding="utf-8"
    )
    (reports / "attribution.json").write_text(
        '[{"dataset":"fixture","license":"Apache-2.0"}]\n', encoding="utf-8"
    )
    (reports / "raw-inventory.json").write_text(
        '[{"dataset_id":"fixture","archive_sha256":"' + "b" * 64 + '"}]\n',
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in [*processed.iterdir(), *reports.iterdir()]
        if path.is_file()
    }
    manifest = {
        "fingerprint": fingerprint,
        "retrieval_count": 1,
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "hashes": hashes,
    }
    (reports / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    state = root / "artifacts" / "state"
    state.mkdir(parents=True)
    (state / "active_preparation.json").write_bytes(
        canonical_json(
            {
                "fingerprint": fingerprint,
                "processed_path": str(processed),
                "report_path": str(reports),
            }
        )
        + b"\n"
    )
    gates = root / "artifacts" / "gates"
    gates.mkdir(parents=True)
    (gates / "gate-b.json").write_text('{"status":"DATA_PREPARED"}\n', encoding="utf-8")
    return fingerprint


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir()
    decompressor = zstandard.ZstdDecompressor()
    with (
        archive.open("rb") as source,
        decompressor.stream_reader(source) as reader,
        tarfile.open(fileobj=reader, mode="r|") as value,
    ):
        value.extractall(destination)


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    compressor = zstandard.ZstdCompressor()
    with (
        path.open("wb") as destination,
        compressor.stream_writer(destination, closefd=False) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as value,
    ):
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            value.addfile(info, io.BytesIO(payload))


def test_bundle_is_deterministic_and_validates_clean_extraction(
    workspace_factory: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = workspace_factory()  # type: ignore[operator]
    fingerprint = _completed_preparation(root)
    monkeypatch.setattr(bundle, "_tracked_files", lambda _workspace: ["pyproject.toml"])
    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    one = bundle.create(root, fingerprint, first)
    two = bundle.create(root, fingerprint, second)
    assert one["sha256"] == two["sha256"]
    assert first.read_bytes() == second.read_bytes()

    extracted = tmp_path / "extracted"
    _extract(first, extracted)
    result = bundle.validate(extracted)
    assert result["valid"] is True
    from acharya.config import Settings
    from acharya.preparation import verify_handoff

    settings = Settings.load(root)
    from dataclasses import replace

    assert verify_handoff(replace(settings, workspace=extracted), strict=True)["verified"]
    paths = {path.relative_to(extracted).as_posix() for path in extracted.rglob("*")}
    assert not any(
        ".env" in path or "raw/" in path or path.endswith(".safetensors") for path in paths
    )
    manifest = json.loads((extracted / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    assert all("sha256" in record and "license_scope" in record for record in manifest["files"])


def test_bundle_refuses_incomplete_preparation(workspace_factory: object, tmp_path: Path) -> None:
    root = workspace_factory()  # type: ignore[operator]
    with pytest.raises(PreflightError, match="completed strict preparation"):
        bundle.create(root, "missing", tmp_path / "bad.tar.zst")


def test_archive_validation_enforces_member_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "oversized.tar.zst"
    _write_archive(archive, [("large.txt", b"1234")])
    monkeypatch.setattr(bundle, "_MAX_MEMBER_BYTES", 3)
    with pytest.raises(PreflightError, match="member-size"):
        bundle.validate_archive(archive)


def test_archive_validation_enforces_member_count_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "many.tar.zst"
    _write_archive(archive, [("one.txt", b"1"), ("two.txt", b"2")])
    monkeypatch.setattr(bundle, "_MAX_MEMBERS", 1)
    with pytest.raises(PreflightError, match="member-count"):
        bundle.validate_archive(archive)


def test_archive_validation_enforces_expanded_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "expansion.tar.zst"
    _write_archive(archive, [("one.txt", b"123"), ("two.txt", b"456")])
    monkeypatch.setattr(bundle, "_MAX_EXPANDED_BYTES", 5)
    with pytest.raises(PreflightError, match="expanded-size"):
        bundle.validate_archive(archive)


def test_runtime_activation_preserves_manifest_and_rejects_source_tampering(
    workspace_factory: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acharya.lightning import verify_bundle

    root = workspace_factory()  # type: ignore[operator]
    fingerprint = _completed_preparation(root)
    monkeypatch.setattr(bundle, "_tracked_files", lambda _workspace: ["pyproject.toml"])
    archive = tmp_path / "runtime.tar.zst"
    bundle.create(root, fingerprint, archive)
    extracted = tmp_path / "runtime"
    _extract(archive, extracted)
    (extracted / ".venv").mkdir()
    (extracted / ".venv" / "installed.txt").write_text("runtime")
    (extracted / "artifacts" / "gates").mkdir()
    (extracted / "artifacts" / "gates" / "gate-c.json").write_text("{}")
    assert bundle.activate(extracted)["activated"] is True
    cache = extracted / ".cache/huggingface/models--fixture/snapshots/revision"
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_bytes(b"runtime model cache")
    assert bundle.activate(extracted)["activated"] is True
    verify_bundle(extracted)
    (extracted / "pyproject.toml").write_text("tampered")
    with pytest.raises(PreflightError, match="hash mismatch"):
        bundle.activate(extracted)
