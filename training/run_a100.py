"""Bounded A100 workflow with an immutable, portable two-epoch training target."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from acharya.bundle import activate
from acharya.config import canonical_json, sha256_file
from acharya.lightning import QLoRAProfile, preflight, run_external_stage


def run(
    workspace: Path,
    hours: float,
    quote: Path | None,
    prepare_only: bool,
    accelerator: str = "auto",
) -> None:
    profile = QLoRAProfile.load(workspace)
    if not 0 < hours <= 8:
        raise ValueError("--hours must be positive and at most 8")
    if accelerator == "H200" and hours > 3:
        raise ValueError("H200 session budget is capped at 3 hours")
    activate(workspace)
    if prepare_only:
        # Run on the CPU machine before starting billed A100 time.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.train_qlora",
                "--workspace",
                str(workspace),
                "--acquire-only",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "acharya.cli",
                "--workspace",
                str(workspace),
                "index",
                "build",
                "--retrieval-mode",
                "full",
            ],
            check=True,
        )
        for arguments in (
            [
                "evaluate",
                "--retrieval-mode",
                "full",
                "--golden",
                str(workspace / "eval/golden.jsonl"),
                "--activate-calibration",
            ],
            ["calibrate-support"],
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "acharya.cli",
                    *arguments,
                    "--workspace",
                    str(workspace),
                ],
                check=True,
            )
        return
    hardware = preflight(workspace, cpu_only_dry_run=False, quote=quote)
    if accelerator != "auto":
        devices = hardware.get("gpu", {}).get("devices", [])
        if len(devices) != 1 or accelerator not in str(devices[0]["name"]):
            raise RuntimeError(f"requested {accelerator} does not match the allocated GPU")
    started = time.monotonic()
    total_seconds = hours * 3600
    reserve = (60 if hours <= 3 else int(profile.external["evaluation_reserve_minutes"])) * 60
    run_root = workspace / "artifacts/external-run"
    run_root.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((workspace / "artifacts/state/active_preparation.json").read_text())
    processed = workspace / pointer["processed_path"]
    row_count = sum(
        bool(line.strip()) for line in (processed / "train.jsonl").read_text().splitlines()
    )
    target_steps = math.ceil(row_count / int(profile.training["effective_batch_size"])) * 2
    plan = {
        "fingerprint": pointer["fingerprint"],
        "profile_sha256": profile.config_sha256,
        "train_sha256": sha256_file(processed / "train.jsonl"),
        "target_steps": target_steps,
        "maximum_epochs": 2,
        "rows": row_count,
    }
    plan_path = run_root / "training-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("existing training plan differs; do not mix runs")
    plan_path.write_bytes(canonical_json(plan) + b"\n")
    print(
        json.dumps(
            {"plan": plan, "session_hours": hours, "evaluation_reserve_minutes": reserve // 60}
        ),
        flush=True,
    )

    def stage(name: str, module: str, *arguments: str) -> None:
        remaining = total_seconds - (time.monotonic() - started)
        if remaining < 60:
            raise RuntimeError("session time budget exhausted; recovery archive preserves progress")
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180s",
            str(int(remaining)),
            sys.executable,
            "-u",
            "-m",
            module,
            *arguments,
            "--workspace",
            str(workspace),
        ]
        run_external_stage(workspace, name, command)

    stage("model_acquisition", "training.train_qlora", "--acquire-only")
    recovery = run_root / "checkpoint"
    if not any(recovery.glob("checkpoint-*/COMPLETE.json")):
        stage(
            "smoke",
            "training.train_qlora",
            "--output",
            str(run_root / "smoke"),
            "--max-steps",
            "25",
        )
        metrics = json.loads((run_root / "smoke/train_metrics.json").read_text())
        seconds_per_step = float(metrics["train_runtime"]) / int(metrics["global_step"])
        remaining = total_seconds - (time.monotonic() - started) - reserve
        print(
            json.dumps(
                {
                    "measured_seconds_per_step": seconds_per_step,
                    "estimated_training_minutes": target_steps * seconds_per_step / 60,
                    "available_training_minutes": remaining / 60,
                    "estimate_is_not_guarantee": True,
                }
            ),
            flush=True,
        )
        if target_steps * seconds_per_step * 1.2 > remaining:
            raise RuntimeError(
                "Measured training estimate plus 20% margin exceeds the training budget. "
                "Stop the paid machine and use a longer authorized session; "
                "no full run was started."
            )
    training_seconds = total_seconds - (time.monotonic() - started) - reserve
    if training_seconds < 180:
        raise RuntimeError("insufficient training time after reserving evaluation/export")
    os.environ["ACHARYA_TRAIN_SECONDS"] = str(int(training_seconds))
    os.environ["ACHARYA_TRAIN_DEADLINE_UNIX"] = str(time.time() + training_seconds)
    stage(
        "training",
        "training.train_qlora",
        "--output",
        str(recovery),
        "--max-steps",
        str(target_steps),
    )
    stage(
        "evaluation",
        "training.evaluate_adapter",
        "--adapter",
        str(recovery / "adapter"),
        "--output",
        str(run_root / "evaluation/evaluation.json"),
    )
    stage(
        "export",
        "training.export_adapter",
        "--source",
        str(recovery / "adapter"),
        "--destination",
        str(run_root / "adapter"),
    )
    from acharya.config import Settings
    from acharya.rag.index import load_index

    try:
        current_index = load_index(Settings.load(workspace), expected_mode="full")
        index_ready = current_index.corpus_fingerprint == pointer["fingerprint"]
    except (OSError, ValueError, RuntimeError):
        index_ready = False
    if not index_ready:
        stage("rag_index", "acharya.cli", "index", "build", "--retrieval-mode", "full")
    stage(
        "rag_calibration",
        "acharya.cli",
        "evaluate",
        "--retrieval-mode",
        "full",
        "--golden",
        str(workspace / "eval/golden.jsonl"),
        "--activate-calibration",
    )
    stage("support_calibration", "acharya.cli", "calibrate-support")
    stage(
        "acceptance",
        "training.acceptance",
        "--adapter",
        str(run_root / "adapter"),
        "--output",
        str(run_root / "evaluation/acceptance.json"),
    )
    stage(
        "rag_smoke",
        "training.peft_rag_smoke",
        "--adapter",
        str(run_root / "adapter"),
        "--output",
        str(run_root / "serving/peft-rag-smoke.json"),
    )
    stage("finalize", "acharya.lightning", "finalize", "--run-root", str(run_root))
    print(
        "COMPLETE: read evaluation/acceptance.json and its human-review checklist before iOS use.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--quote", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "A100", "H200"], default="auto")
    args = parser.parse_args()
    run(args.workspace.resolve(), args.hours, args.quote, args.prepare_only, args.accelerator)


if __name__ == "__main__":
    main()
