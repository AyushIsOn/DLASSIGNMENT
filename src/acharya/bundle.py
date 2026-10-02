"""Deterministic, secret-free Lightning transfer bundle creation and validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard

from acharya.config import canonical_json, sha256_file
from acharya.lightning import PreflightError, _preparation_state

_TOP_LEVEL = {
    ".python-version",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "LIGHTNING_HANDOFF.md",
    ".gitleaks.toml",
    "README.md",
    "data/dataset V1.0.pdf",
    "pyproject.toml",
    "uv.lock",
}
_PREFIXES = (
    ".github/",
    "AcharyaGPT(iOS)/",
    "configs/",
    "eval/",
    "prompts/",
    "scripts/",
    "src/",
    "tests/",
    "training/",
)
_FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "adapters",
    "checkpoints",
    "model-cache",
    "models",
    "indexes",
    "provider-check",
    "raw",
}
_FORBIDDEN_SUFFIXES = (".key", ".pem", ".safetensors", ".gguf", ".pyc")
_MAX_ARCHIVE_BYTES = 2 * 1024**3
_MAX_EXPANDED_BYTES = 8 * 1024**3
_MAX_MEMBER_BYTES = 2 * 1024**3
_MAX_MEMBERS = 100_000


def _safe_relative(relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise PreflightError(f"unsafe bundle path:{relative}")
    if any(part in _FORBIDDEN_PARTS for part in value.parts):
        raise PreflightError(f"forbidden bundle path:{relative}")
    if any(part.startswith(".env") for part in value.parts):
        raise PreflightError(f"dotenv files cannot be bundled:{relative}")
    if relative.endswith(_FORBIDDEN_SUFFIXES):
        raise PreflightError(f"forbidden bundle artifact:{relative}")
    return Path(*value.parts)


def _tracked_files(workspace: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "-z"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return sorted(item.decode() for item in result.stdout.split(b"\0") if item)


def _selected_tracked(workspace: Path) -> list[tuple[Path, str]]:
    selected = []
    for relative in _tracked_files(workspace):
        if relative not in _TOP_LEVEL and not relative.startswith(_PREFIXES):
            continue
        safe = _safe_relative(relative)
        source = workspace / safe
        if source.is_file() and not source.is_symlink():
            selected.append((source, safe.as_posix()))
    return selected


def _preparation_files(workspace: Path, preparation: dict[str, object]) -> list[tuple[Path, str]]:
    selected = []
    fingerprint = str(preparation["fingerprint"])
    for key, prefix in (
        ("processed_path", f"data/processed/{fingerprint}"),
        ("report_path", f"artifacts/preparation/{fingerprint}"),
    ):
        root = Path(str(preparation[key]))
        for source in sorted(root.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            relative = f"{prefix}/{source.relative_to(root).as_posix()}"
            selected.append((source, _safe_relative(relative).as_posix()))
    return selected


def _copy_files(stage: Path, selected: list[tuple[Path, str]]) -> None:
    for source, relative in selected:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o755 if relative.endswith(".sh") else 0o644)


def _write_metadata(stage: Path, preparation: dict[str, object]) -> None:
    raw_inventory_path = (
        stage / "artifacts" / "preparation" / str(preparation["fingerprint"]) / "raw-inventory.json"
    )
    raw_inventory: object = []
    if raw_inventory_path.is_file():
        raw_inventory = json.loads(raw_inventory_path.read_text(encoding="utf-8"))
    (stage / "COPY_UPLOAD_INVENTORY.json").write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "copy": ["the single .tar.zst bundle"],
                "upload": ["persistent Lightning workspace storage"],
                "do_not_upload": [
                    "credentials or dotenv files",
                    "raw Kaggle archives",
                    "model caches, indexes, adapters, or checkpoints",
                ],
                "raw_inventory_hashes_only": raw_inventory,
            }
        )
        + b"\n"
    )
    state = stage / "artifacts" / "state"
    state.mkdir(parents=True, exist_ok=True)
    fingerprint = str(preparation["fingerprint"])
    pointer_value = {
        "fingerprint": fingerprint,
        "processed_path": f"data/processed/{fingerprint}",
        "report_path": f"artifacts/preparation/{fingerprint}",
    }
    (state / "active_preparation.json").write_bytes(canonical_json(pointer_value) + b"\n")
    (state / "active_corpus.json").write_bytes(
        canonical_json({"fingerprint": fingerprint, "path": f"data/processed/{fingerprint}"})
        + b"\n"
    )
    (stage / "BOOTSTRAP.md").write_text(
        "# Lightning bootstrap\n\n"
        "Install with `uv sync --frozen --extra retrieval --extra training --group dev`, "
        "then verify with `uv run python -m acharya.bundle validate --workspace $PWD`. "
        "then run `bash scripts/lightning_a100.sh --workspace $PWD --quote quote.json`. "
        "Keep quote evidence, persistent storage, and auto-stop enabled. "
        "GPU training is optional.\n",
        encoding="utf-8",
    )


def _manifest(stage: Path, preparation: dict[str, object]) -> dict[str, Any]:
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "BUNDLE_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "license_scope": "see artifacts/preparation attribution or repository LICENSE",
                }
            )
    return {
        "schema_version": 1,
        "format": "tar+zstd",
        "source_date_epoch": 0,
        "preparation_fingerprint": preparation["fingerprint"],
        "preparation_manifest_sha256": preparation["manifest_sha256"],
        "files": files,
        "exclusions": sorted(_FORBIDDEN_PARTS),
    }


def _archive(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    compressor = zstandard.ZstdCompressor(level=19, threads=0, write_checksum=True)
    with (
        temporary.open("wb") as destination,
        compressor.stream_writer(destination, closefd=False) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        paths = sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix())
        for path in paths:
            relative = path.relative_to(stage).as_posix()
            info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() or relative.endswith(".sh") else 0o644
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = path.stat().st_size
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    os.replace(temporary, output)


def create(workspace: Path, fingerprint: str, output: Path) -> dict[str, object]:
    preparation = _preparation_state(workspace)
    if not preparation.get("ready"):
        raise PreflightError("bundle requires a completed strict preparation")
    if preparation.get("fingerprint") != fingerprint:
        raise PreflightError("requested bundle fingerprint is not the active strict preparation")
    gate_path = workspace / "artifacts" / "gates" / "gate-b.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "BLOCKED_EXTERNAL_DATA":
            raise PreflightError("Gate B external data blocker prevents bundle creation")
    selected = _selected_tracked(workspace) + _preparation_files(workspace, preparation)
    with tempfile.TemporaryDirectory(prefix="acharya-bundle-") as temporary:
        stage = Path(temporary)
        _copy_files(stage, selected)
        _write_metadata(stage, preparation)
        manifest = _manifest(stage, preparation)
        (stage / "BUNDLE_MANIFEST.json").write_bytes(canonical_json(manifest) + b"\n")
        _archive(stage, output)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "size": output.stat().st_size,
        "fingerprint": fingerprint,
        "file_count": len(manifest["files"]),
    }


def _runtime_file(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return "__pycache__" in parts or name.startswith(
        (
            ".venv/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            "artifacts/gates/",
            "artifacts/external-run/",
            "artifacts/model-cache/",
            "artifacts/indexes/",
            "artifacts/calibration/",
            "artifacts/state/",
        )
    )


def validate(workspace: Path, *, runtime: bool = False) -> dict[str, object]:
    manifest_path = workspace / "BUNDLE_MANIFEST.json"
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["files"]
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise PreflightError("bundle manifest is unavailable") from error
    expected = {str(record["path"]): record for record in records}
    actual = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path.name != "BUNDLE_MANIFEST.json"
    }
    if runtime:
        actual = {name for name in actual if name in expected or not _runtime_file(name)}
    if actual != set(expected):
        raise PreflightError("bundle extraction membership does not match its manifest")
    for relative, record in expected.items():
        _safe_relative(relative)
        path = workspace / relative
        if path.stat().st_size != int(record["size"]) or sha256_file(path) != record["sha256"]:
            raise PreflightError(f"bundle extraction hash mismatch:{relative}")
    return {
        "valid": True,
        "fingerprint": manifest["preparation_fingerprint"],
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(expected),
    }


def validate_archive(archive: Path) -> dict[str, object]:
    """Validate a size-bounded archive after a safe clean extraction."""
    if not archive.is_file() or archive.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise PreflightError("bundle compressed-size limit exceeded")
    with tempfile.TemporaryDirectory(prefix="acharya-validate-") as temporary:
        destination = Path(temporary)
        decompressor = zstandard.ZstdDecompressor()
        member_count = 0
        expanded_bytes = 0
        seen: set[str] = set()
        with (
            archive.open("rb") as source,
            decompressor.stream_reader(source) as reader,
            tarfile.open(fileobj=reader, mode="r|") as value,
        ):
            for member in value:
                member_count += 1
                if member_count > _MAX_MEMBERS:
                    raise PreflightError("bundle member-count limit exceeded")
                relative = PurePosixPath(member.name.rstrip("/"))
                if not relative.parts:
                    continue
                relative_name = relative.as_posix()
                _safe_relative(relative_name)
                if relative_name in seen:
                    raise PreflightError("bundle contains a duplicate member")
                seen.add(relative_name)
                supported = member.isdir() or member.isfile()
                if member.issym() or member.islnk() or not supported:
                    raise PreflightError("bundle contains a link or special member")
                if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
                    raise PreflightError("bundle member-size limit exceeded")
                expanded_bytes += member.size
                if expanded_bytes > _MAX_EXPANDED_BYTES:
                    raise PreflightError("bundle expanded-size limit exceeded")
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = value.extractfile(member)
                if source_file is None:
                    raise PreflightError(f"bundle member cannot be read:{member.name}")
                written = 0
                with target.open("wb") as output:
                    while block := source_file.read(1024 * 1024):
                        written += len(block)
                        if written > member.size or written > _MAX_MEMBER_BYTES:
                            raise PreflightError("bundle member-size limit exceeded")
                        output.write(block)
                if written != member.size:
                    raise PreflightError("bundle member size does not match its header")
        result = validate(destination)
    return {**result, "archive_sha256": sha256_file(archive)}


def activate(workspace: Path) -> dict[str, object]:
    """Validate portable relative pointers without rewriting hash-bound bundle inputs."""
    result = validate(workspace, runtime=True)
    return {**result, "activated": True}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m acharya.bundle")
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create")
    build.add_argument("--workspace", type=Path, required=True)
    build.add_argument("--fingerprint", required=True)
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--workspace", type=Path, required=True)
    archive_check = commands.add_parser("validate-archive")
    archive_check.add_argument("--archive", type=Path, required=True)
    relocate = commands.add_parser("activate")
    relocate.add_argument("--workspace", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "create":
        result = create(args.workspace.resolve(), args.fingerprint, args.output.resolve())
    elif args.command == "validate":
        result = validate(args.workspace.resolve(), runtime=True)
    elif args.command == "validate-archive":
        result = validate_archive(args.archive.resolve())
    else:
        result = activate(args.workspace.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
