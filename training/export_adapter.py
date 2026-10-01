"""Validate and atomically export a PEFT adapter with immutable hashes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from acharya.config import canonical_json, sha256_file
from acharya.lightning import LOCKED_REPOSITORY, LOCKED_REVISION, PreflightError, QLoRAProfile

REQUIRED = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json")


def export_adapter(workspace: Path, source: Path, destination: Path) -> dict[str, object]:
    profile = QLoRAProfile.load(workspace)
    for name in REQUIRED:
        if not (source / name).is_file():
            raise PreflightError(f"required adapter file missing:{name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".adapter-export-", dir=destination.parent))
    try:
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            if path.is_file() and not path.is_symlink():
                shutil.copy2(path, stage / path.name)
        files = {
            path.name: sha256_file(path)
            for path in sorted(stage.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "base_repository": LOCKED_REPOSITORY,
            "base_revision": LOCKED_REVISION,
            "qlora_config_sha256": profile.config_sha256,
            "files": files,
            "atomic_export": True,
        }
        (stage / "ADAPTER_MANIFEST.json").write_bytes(canonical_json(manifest) + b"\n")
        if destination.exists():
            existing_manifest = destination / "ADAPTER_MANIFEST.json"
            if existing_manifest.is_file() and existing_manifest.read_bytes() == (
                stage / "ADAPTER_MANIFEST.json"
            ).read_bytes():
                shutil.rmtree(stage)
                return manifest
            raise PreflightError("adapter destination already exists with different hashes")
        os.replace(stage, destination)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_adapter(
                args.workspace.resolve(), args.source.resolve(), args.destination.resolve()
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
