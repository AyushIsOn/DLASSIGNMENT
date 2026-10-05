from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import run_a100


@pytest.mark.parametrize(
    "hours,accelerator,seconds_per_step,budget_failure",
    [
        (7, "auto", 1, False),
        (3, "H200", 1, False),
        (2, "H200", 1, False),
        (2, "H200", 100, True),
    ],
)
def test_runner_uses_immutable_target_and_valid_cli_order(
    workspace_factory: object,
    monkeypatch: pytest.MonkeyPatch,
    hours: int,
    accelerator: str,
    seconds_per_step: int,
    budget_failure: bool,
) -> None:
    root = workspace_factory()  # type: ignore[operator]
    (root / "artifacts/state").mkdir(parents=True)
    (root / "data/processed").mkdir()
    (root / "data/processed/train.jsonl").write_text("{}\n" * 2093)
    (root / "artifacts/state/active_preparation.json").write_text(
        json.dumps(
            {
                "fingerprint": "fixture",
                "processed_path": "data/processed",
            }
        )
    )
    monkeypatch.setattr(run_a100, "activate", lambda _: {})
    monkeypatch.setattr(
        run_a100,
        "preflight",
        lambda *args, **kwargs: {"gpu": {"devices": [{"name": "NVIDIA H200"}]}},
    )
    calls = []

    def stage(workspace: Path, name: str, command: list[str]) -> dict[str, object]:
        calls.append((name, command))
        assert command[-2:] == ["--workspace", str(workspace)]
        module = command[command.index("-m") + 1]
        arguments = command[command.index("-m") + 2 :]
        if module == "acharya.cli":
            from acharya.cli import parser

            parser().parse_args(arguments)
        elif module == "acharya.lightning":
            from acharya.lightning import parser

            parser().parse_args(arguments)
        if name == "smoke":
            output = workspace / "artifacts/external-run/smoke"
            output.mkdir(parents=True)
            (output / "train_metrics.json").write_text(
                json.dumps(
                    {
                        "train_runtime": 25 * seconds_per_step,
                        "global_step": 25,
                    }
                )
            )
        return {}

    monkeypatch.setattr(run_a100, "run_external_stage", stage)
    if budget_failure:
        with pytest.raises(RuntimeError, match="exceeds the training budget"):
            run_a100.run(root, hours, None, False, accelerator)
        assert "training" not in {name for name, _ in calls}
        return
    run_a100.run(root, hours, None, False, accelerator)
    plan = json.loads((root / "artifacts/external-run/training-plan.json").read_text())
    assert plan["target_steps"] == 262
    assert {name for name, _ in calls} >= {"training", "acceptance", "finalize", "rag_index"}
    assert next(command for name, command in calls if name == "training")[-4:-2] == [
        "--max-steps",
        "262",
    ]
    with pytest.raises(ValueError, match="at most 8"):
        run_a100.run(root, 9, None, False)


@pytest.mark.parametrize(
    "gpu_csv,expected",
    [
        ("NVIDIA H200, 143771", True),
        ("NVIDIA A100-SXM4-80GB, 81920", True),
        ("NVIDIA A100-SXM4-40GB, 40960", False),
        ("NVIDIA H200, 143771\nNVIDIA H200, 143771", False),
        ("NVIDIA T4, 15360", False),
    ],
)
def test_hardware_preflight_recognizes_only_supported_single_gpus(
    monkeypatch: pytest.MonkeyPatch,
    gpu_csv: str,
    expected: bool,
) -> None:
    from types import SimpleNamespace

    from acharya import lightning

    monkeypatch.setattr(lightning.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        lightning.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=gpu_csv)
    )
    assert lightning._gpu_snapshot()["compatible"] is expected
