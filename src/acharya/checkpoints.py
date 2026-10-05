"""Completed, hash-verified training checkpoints and portable recovery archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from acharya.config import canonical_json, sha256_file

_REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "RUN_INPUT.json",
}


def seal(path: Path) -> dict[str, Any]:
    missing = _REQUIRED - {p.name for p in path.iterdir() if p.is_file()}
    if missing:
        raise RuntimeError(f"incomplete resumable checkpoint: {sorted(missing)}")
    binding = json.loads((path / "RUN_INPUT.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    step = int(state["global_step"])
    if path.name != f"checkpoint-{step}":
        raise RuntimeError("checkpoint step/name mismatch")
    files = {
        p.name: sha256_file(p)
        for p in sorted(path.iterdir())
        if p.is_file() and p.name not in {"COMPLETE.json", ".COMPLETE.tmp"}
    }
    value = {"step": step, "run_sha256": binding["run_sha256"], "files": files}
    temporary = path / ".COMPLETE.tmp"
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path / "COMPLETE.json")
    return value


def verify(path: Path, run_sha256: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((path / "COMPLETE.json").read_text())
    files = value["files"]
    if not isinstance(files, dict) or not _REQUIRED.issubset(files):
        raise RuntimeError("checkpoint lacks optimizer/scheduler/RNG state")
    if run_sha256 is not None and value["run_sha256"] != run_sha256:
        raise RuntimeError("checkpoint belongs to different training inputs")
    for name, digest in files.items():
        if Path(name).name != name or (path / name).is_symlink():
            raise RuntimeError("unsafe checkpoint path")
        if sha256_file(path / name) != digest:
            raise RuntimeError(f"checkpoint checksum mismatch: {name}")
    if json.loads((path / "RUN_INPUT.json").read_text())["run_sha256"] != value["run_sha256"]:
        raise RuntimeError("checkpoint binding mismatch")
    if int(json.loads((path / "trainer_state.json").read_text())["global_step"]) != value["step"]:
        raise RuntimeError("checkpoint state mismatch")
    return value


def pack(workspace: Path, checkpoint: Path, output: Path) -> dict[str, object]:
    value = verify(checkpoint)
    checkpoints = {checkpoint.name: checkpoint}
    best = json.loads((checkpoint / "trainer_state.json").read_text()).get("best_model_checkpoint")
    if best:
        best_path = checkpoint.parent / Path(best).name
        verify(best_path, value["run_sha256"])
        checkpoints[best_path.name] = best_path
    pointer = json.loads((workspace / "artifacts/state/active_preparation.json").read_text())
    metadata = {
        "checkpoints": sorted(checkpoints),
        "checkpoint": checkpoint.name,
        "fingerprint": pointer["fingerprint"],
        "profile_sha256": sha256_file(workspace / "configs/qlora.yaml"),
        "run_sha256": value["run_sha256"],
        "step": value["step"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with tarfile.open(temporary, "w") as archive:
            import io

            payload = canonical_json(metadata)
            info = tarfile.TarInfo("RECOVERY.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for folder in checkpoints.values():
                manifest = verify(folder, value["run_sha256"])
                for name in [*manifest["files"], "COMPLETE.json"]:
                    archive.add(folder / name, arcname=f"{folder.name}/{name}", recursive=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **metadata,
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }


def restore(workspace: Path, archive_path: Path) -> dict[str, object]:
    target_root = workspace / "artifacts/external-run/checkpoint"
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=target_root) as tmp:
        stage = Path(tmp)
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            if len(members) > 100 or sum(m.size for m in members) > 8 * 1024**3:
                raise RuntimeError("recovery archive exceeds limits")
            for member in members:
                path = Path(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) > 2
                ):
                    raise RuntimeError("unsafe recovery archive member")
            if len({m.name for m in members}) != len(members):
                raise RuntimeError("duplicate recovery archive member")
            archive.extractall(stage, filter="data")
        metadata = json.loads((stage / "RECOVERY.json").read_text())
        name = str(metadata["checkpoint"])
        if Path(name).name != name or not name.startswith("checkpoint-"):
            raise RuntimeError("invalid recovery checkpoint")
        pointer = json.loads((workspace / "artifacts/state/active_preparation.json").read_text())
        if metadata["fingerprint"] != pointer["fingerprint"] or metadata[
            "profile_sha256"
        ] != sha256_file(workspace / "configs/qlora.yaml"):
            raise RuntimeError("recovery data/model configuration differs from this workspace")
        names = metadata.get("checkpoints", [name])
        if not isinstance(names, list) or name not in names or len(names) > 2:
            raise RuntimeError("invalid recovery checkpoint inventory")
        for item in names:
            if (
                not isinstance(item, str)
                or Path(item).name != item
                or not item.startswith("checkpoint-")
            ):
                raise RuntimeError("invalid recovery checkpoint path")
            verify(stage / item, str(metadata["run_sha256"]))
        # Rewrite absolute Trainer paths only after checking the original hashes.
        for item in names:
            state_path = stage / item / "trainer_state.json"
            state = json.loads(state_path.read_text())
            best = state.get("best_model_checkpoint")
            if best:
                best_name = Path(best).name
                # An older best snapshot can reference an even older best; only the
                # latest Trainer state is resumed and must resolve inside this archive.
                if item == name and best_name not in names:
                    raise RuntimeError("archive omits best checkpoint")
                state["best_model_checkpoint"] = str(target_root / best_name)
                state_path.write_bytes(canonical_json(state) + b"\n")
                seal(stage / item)
            destination = target_root / item
            if destination.exists() and verify(destination) != verify(stage / item):
                raise RuntimeError("refusing to overwrite a different checkpoint")
        for item in names:
            if not (target_root / item).exists():
                shutil.move(str(stage / item), target_root / item)
        return {"restored": True, "checkpoint": str(target_root / name), "step": metadata["step"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pack", "restore"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.action == "pack":
        if args.checkpoint is None:
            parser.error("pack requires --checkpoint")
        result = pack(workspace, args.checkpoint.resolve(), args.archive.resolve())
    else:
        result = restore(workspace, args.archive.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
