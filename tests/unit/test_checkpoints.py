from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from acharya.checkpoints import pack, restore, seal, verify
from acharya.lightning import select_resume_checkpoint


def checkpoint(root: Path, step: int, best: Path | None = None) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (path / name).write_bytes(f"fixture-{step}-{name}".encode())
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "best_model_checkpoint": str(best) if best else None,
            }
        )
    )
    (path / "RUN_INPUT.json").write_text(json.dumps({"run_sha256": "run"}))
    seal(path)
    return path


def workspace(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs/qlora.yaml").write_text("same profile")
    (root / "artifacts/state").mkdir(parents=True)
    (root / "artifacts/state/active_preparation.json").write_text('{"fingerprint":"same"}')


def test_corrupt_latest_falls_back_to_completed_checkpoint(tmp_path: Path) -> None:
    older = checkpoint(tmp_path, 25)
    latest = checkpoint(tmp_path, 50, older)
    (latest / "optimizer.pt").write_bytes(b"incomplete write")
    with pytest.raises(RuntimeError, match="checksum"):
        verify(latest)
    assert select_resume_checkpoint(tmp_path, "run") == older
    assert select_resume_checkpoint(tmp_path, "other run") is None


def test_transfer_preserves_best_optimizer_rng_and_relocates_paths(tmp_path: Path) -> None:
    first, second = tmp_path / "account-a", tmp_path / "account-b"
    workspace(first)
    workspace(second)
    older = checkpoint(first / "artifacts/external-run/checkpoint", 25)
    latest = checkpoint(older.parent, 50, older)
    archive = tmp_path / "recovery.tar"
    pack(first, latest, archive)
    result = restore(second, archive)
    destination = Path(str(result["checkpoint"]))
    state = json.loads((destination / "trainer_state.json").read_text())
    best = Path(state["best_model_checkpoint"])
    assert best == destination.parent / "checkpoint-25"
    verify(best, "run")
    verify(destination, "run")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        assert (destination / name).read_bytes() == (latest / name).read_bytes()
    assert restore(second, archive)["restored"]  # idempotent
    (second / "configs/qlora.yaml").write_text("changed")
    with pytest.raises(RuntimeError, match="differs"):
        restore(second, archive)


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    workspace(tmp_path / "target")
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../escape")
        handle.addfile(info)
    with pytest.raises(RuntimeError, match="unsafe"):
        restore(tmp_path / "target", archive)
