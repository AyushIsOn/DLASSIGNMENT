"""CPU-verifiable orchestration for the optional external QLoRA workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acharya.config import canonical_json, load_yaml, sha256_bytes, sha256_file

LOCKED_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct"
LOCKED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class PreflightError(RuntimeError):
    """Raised before paid model loading when an external prerequisite is unsafe."""


@dataclass(frozen=True)
class QLoRAProfile:
    workspace: Path
    value: dict[str, Any]
    config_sha256: str

    @classmethod
    def load(cls, workspace: Path | str) -> QLoRAProfile:
        root = Path(workspace).expanduser().resolve()
        path = root / "configs" / "qlora.yaml"
        value = load_yaml(path)
        base = value.get("base_model", {})
        quantization = value.get("quantization", {})
        lora = value.get("lora", {})
        training = value.get("training", {})
        external = value.get("external", {})
        expected: tuple[tuple[object, object, str], ...] = (
            (base.get("repository"), LOCKED_REPOSITORY, "base repository"),
            (base.get("revision"), LOCKED_REVISION, "base revision"),
            (base.get("trust_remote_code"), False, "trust_remote_code"),
            (quantization.get("quant_type"), "nf4", "quantization type"),
            (quantization.get("compute_dtype"), "bfloat16", "compute dtype"),
            (lora.get("rank"), 16, "LoRA rank"),
            (training.get("maximum_sequence_tokens"), 1536, "sequence length"),
            (training.get("effective_batch_size"), 16, "effective batch"),
            (training.get("cpu_dry_run_steps"), 5, "CPU dry-run steps"),
            (training.get("external_smoke_steps"), 100, "external smoke steps"),
            (external.get("maximum_credits"), 10, "credit cap"),
            (external.get("maximum_wall_minutes"), 240, "wall cap"),
            (external.get("minimum_vram_gib"), 39, "VRAM minimum"),
        )
        for actual, locked, label in expected:
            if actual != locked:
                raise PreflightError(f"QLoRA {label} must remain locked to {locked!r}")
        if not quantization.get("load_in_4bit") or not quantization.get("double_quant"):
            raise PreflightError("QLoRA must use double-quantized 4-bit loading")
        if int(training.get("micro_batch_size", 0)) * int(
            training.get("gradient_accumulation_steps", 0)
        ) != int(training["effective_batch_size"]):
            raise PreflightError("effective batch size does not reconcile")
        files = base.get("files")
        if not isinstance(files, dict) or len(files) != 4:
            raise PreflightError("all four base weight hashes are required")
        for name, record in files.items():
            if not isinstance(record, dict) or len(str(record.get("sha256", ""))) != 64:
                raise PreflightError(f"invalid locked model hash:{name}")
        stages = value.get("stage_minutes", {})
        locked_stages = {
            "preflight": 10,
            "model_acquisition": 30,
            "smoke": 20,
            "profile": 45,
            "evaluation": 45,
            "export": 30,
            "merge": 45,
        }
        if any(stages.get(name) != minutes for name, minutes in locked_stages.items()):
            raise PreflightError("external stage ceilings do not match the locked profile")
        return cls(root, value, sha256_file(path))

    @property
    def training(self) -> Mapping[str, Any]:
        return self.value["training"]

    @property
    def external(self) -> Mapping[str, Any]:
        return self.value["external"]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def _preparation_state(workspace: Path) -> dict[str, object]:
    pointer = workspace / "artifacts" / "state" / "active_preparation.json"
    gate = workspace / "artifacts" / "gates" / "gate-b.json"
    if not pointer.is_file():
        category = "missing_strict_preparation"
        if gate.is_file():
            try:
                status = json.loads(gate.read_text(encoding="utf-8")).get("status")
                if status == "BLOCKED_EXTERNAL_DATA":
                    category = "blocked_external_data"
            except (OSError, ValueError, TypeError):
                pass
        return {"ready": False, "category": category}
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
        processed = Path(str(value["processed_path"]))
        reports = Path(str(value["report_path"]))
        processed = (processed if processed.is_absolute() else workspace / processed).resolve()
        reports = (reports if reports.is_absolute() else workspace / reports).resolve()
        for path in (processed, reports):
            path.relative_to(workspace)
        manifest_path = reports / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != value.get("fingerprint"):
            raise PreflightError("strict preparation fingerprint mismatch")
        hashes = manifest.get("hashes")
        if not isinstance(hashes, dict):
            raise PreflightError("strict preparation hashes are missing")
        for name, expected in hashes.items():
            path = processed / str(name)
            if not path.is_file():
                path = reports / str(name)
            if not path.is_file() or sha256_file(path) != expected:
                raise PreflightError(f"strict preparation hash mismatch:{name}")
        for split in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            if not (processed / split).is_file() or (processed / split).stat().st_size == 0:
                raise PreflightError(f"strict preparation split missing:{split}")
        return {
            "ready": True,
            "fingerprint": str(value["fingerprint"]),
            "manifest_sha256": sha256_file(manifest_path),
            "processed_path": str(processed),
            "report_path": str(reports),
        }
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise PreflightError("strict preparation evidence is invalid") from error


def cpu_dry_run(steps: int = 5) -> dict[str, object]:
    """Run a deterministic scalar optimization without model access or checkpoints."""
    if steps != 5:
        raise PreflightError("the locked CPU dry-run is exactly five steps")
    weight = 0.0
    losses: list[float] = []
    for _ in range(steps):
        error = weight - 1.0
        losses.append(round(error * error, 12))
        weight -= 0.2 * (2.0 * error)
    return {
        "runner": "deterministic_scalar_cpu_v1",
        "steps": steps,
        "losses": losses,
        "final_weight": round(weight, 12),
        "saved_checkpoint": False,
    }


def fake_optimizer_run(steps: int = 100, *, fail_at: int | None = None) -> dict[str, object]:
    """Exercise bounded optimizer-step accounting without loading a model."""
    completed = 0
    accumulator = 0
    for step in range(1, steps + 1):
        if fail_at == step:
            raise RuntimeError(f"injected fake runner failure at step {step}")
        accumulator = (accumulator * 33 + step) % 1_000_003
        completed = step
    return {"requested_steps": steps, "completed_steps": completed, "digest": accumulator}


class StageStore:
    """Persist input-bound stage markers; failed work is never resumable as complete."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        name: str,
        inputs: Mapping[str, object],
        action: Callable[[], Mapping[str, Path]],
    ) -> dict[str, object]:
        input_sha256 = sha256_bytes(canonical_json(dict(inputs)))
        marker = self.root / f"{name}.json"
        if marker.is_file():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETED" and existing.get(
                "input_sha256"
            ) == input_sha256:
                outputs = existing.get("outputs", {})
                if isinstance(outputs, dict) and all(
                    Path(str(record["path"])).is_file()
                    and sha256_file(Path(str(record["path"]))) == record["sha256"]
                    for record in outputs.values()
                    if isinstance(record, dict)
                ):
                    return existing
        started = time.monotonic()
        try:
            output_paths = action()
            outputs = {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in sorted(output_paths.items())
            }
            result: dict[str, object] = {
                "schema_version": 1,
                "stage": name,
                "status": "COMPLETED",
                "input_sha256": input_sha256,
                "outputs": outputs,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            _atomic_json(marker, result)
            return result
        except BaseException as error:
            _atomic_json(
                marker,
                {
                    "schema_version": 1,
                    "stage": name,
                    "status": "FAILED",
                    "input_sha256": input_sha256,
                    "error_category": type(error).__name__,
                },
            )
            raise


def select_resume_checkpoint(checkpoints: Path, run_sha256: str) -> Path | None:
    """Select only the highest checkpoint bound to the current run input hash."""
    valid: list[tuple[int, Path]] = []
    if not checkpoints.is_dir():
        return None
    for path in checkpoints.glob("checkpoint-*"):
        state_path = path / "trainer_state.json"
        binding_path = path / "RUN_INPUT.json"
        if not state_path.is_file() or not binding_path.is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, TypeError):
            continue
        if binding.get("run_sha256") == run_sha256 and int(state.get("global_step", -1)) == step:
            valid.append((step, path))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]


def _gpu_snapshot() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"compatible": False, "category": "nvidia_smi_unavailable"}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        parsed: list[dict[str, str | int]] = []
        for row in rows:
            name, memory = (item.strip() for item in row.rsplit(",", 1))
            parsed.append({"name": name, "vram_mib": int(memory)})
        compatible = len(parsed) == 1 and "A100" in str(parsed[0]["name"]) and int(
            parsed[0]["vram_mib"]
        ) >= 39 * 1024
        return {"compatible": compatible, "devices": parsed}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"compatible": False, "category": "gpu_query_failed"}


def validate_quote(profile: QLoRAProfile, quote_path: Path) -> dict[str, object]:
    """Validate real currency conversion and abort projections before model loading."""
    try:
        quote = json.loads(quote_path.read_text(encoding="utf-8"))
        credits_per_hour = float(quote["credits_per_hour"])
        currency_per_credit = float(quote["currency_per_credit"])
        startup_minutes = float(quote["startup_minutes"])
        remaining_training_minutes = float(quote["remaining_training_minutes"])
        evaluation_export_minutes = float(quote["evaluation_export_minutes"])
        currency = str(quote["currency"]).strip()
        quoted_at = str(quote["quoted_at_utc"]).strip()
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise PreflightError(
            "quote evidence is missing required numeric conversion fields"
        ) from error
    if not all(
        math.isfinite(value) and value > 0
        for value in (credits_per_hour, currency_per_credit)
    ) or not all(
        math.isfinite(value) and value >= 0
        for value in (startup_minutes, remaining_training_minutes, evaluation_export_minutes)
    ):
        raise PreflightError("quote evidence contains invalid values")
    if not currency or not quoted_at:
        raise PreflightError("quote currency and timestamp are required")
    wall_minutes = startup_minutes + remaining_training_minutes + evaluation_export_minutes
    projected_credits = wall_minutes * credits_per_hour / 60.0
    if wall_minutes > float(profile.external["maximum_wall_minutes"]):
        raise PreflightError("projected wall time exceeds the 240-minute cap")
    if projected_credits > float(profile.external["maximum_credits"]):
        raise PreflightError("projected usage exceeds the 10-credit cap")
    return {
        "quote_sha256": sha256_file(quote_path),
        "currency": currency,
        "currency_per_credit": currency_per_credit,
        "projected_credits": round(projected_credits, 6),
        "projected_currency": round(projected_credits * currency_per_credit, 6),
        "projected_wall_minutes": round(wall_minutes, 6),
    }


def verify_bundle(workspace: Path) -> dict[str, object]:
    manifest_path = workspace / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise PreflightError("bundle manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise PreflightError("bundle manifest is invalid") from error
    if not isinstance(files, list):
        raise PreflightError("bundle file inventory is invalid")
    for record in files:
        relative = Path(str(record["path"]))
        path = (workspace / relative).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError as error:
            raise PreflightError("bundle manifest path escapes workspace") from error
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise PreflightError(f"bundle member missing:{relative}")
        if sha256_file(path) != record["sha256"]:
            raise PreflightError(f"bundle member hash mismatch:{relative}")
    return {"manifest_sha256": sha256_file(manifest_path), "file_count": len(files)}


def preflight(
    workspace: Path, *, cpu_only_dry_run: bool, quote: Path | None = None
) -> dict[str, object]:
    profile = QLoRAProfile.load(workspace)
    preparation = _preparation_state(profile.workspace)
    dry = cpu_dry_run(int(profile.training["cpu_dry_run_steps"]))
    gpu = _gpu_snapshot()
    report: dict[str, object] = {
        "schema_version": 1,
        "gate": "C",
        "status": "PENDING_EXTERNAL_GPU",
        "checked_at": datetime.now(UTC).isoformat(),
        "config_sha256": profile.config_sha256,
        "base_repository": LOCKED_REPOSITORY,
        "base_revision": LOCKED_REVISION,
        "preparation": preparation,
        "cpu_dry_run": dry,
        "gpu": gpu,
        "paid_training_started": False,
        "checkpoint_claimed": False,
        "quality_gain_claimed": False,
    }
    if not cpu_only_dry_run:
        if not bool(preparation.get("ready")):
            raise PreflightError("completed strict preparation is required externally")
        if not bool(gpu.get("compatible")):
            raise PreflightError("exactly one A100 with at least 39 GiB VRAM is required")
        if quote is None:
            raise PreflightError("quote evidence is required")
        report["quote"] = validate_quote(profile, quote)
        minimum_free = int(profile.external["minimum_free_disk_gib"]) * 2**30
        if shutil.disk_usage(workspace).free < minimum_free:
            raise PreflightError("persistent workspace has insufficient free disk")
        if os.environ.get("ACHARYA_PERSISTENT_STORAGE") != "1":
            raise PreflightError("persistent storage confirmation is required")
        if os.environ.get("ACHARYA_AUTO_STOP") != "1":
            raise PreflightError("auto-stop confirmation is required")
        report["bundle"] = verify_bundle(workspace)
        report["external_ready"] = True
    _atomic_json(workspace / "artifacts" / "gates" / "gate-c.json", report)
    return report


def run_external_stage(
    workspace: Path, name: str, command: Sequence[str]
) -> dict[str, object]:
    """Execute one external command through an input-bound, fail-closed stage marker."""
    if not command:
        raise PreflightError("stage command is missing")
    evidence_root = workspace / "artifacts" / "external-run" / "stage-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, object] = {
        "command": list(command),
        "environment_contract": {
            "persistent_storage": os.environ.get("ACHARYA_PERSISTENT_STORAGE"),
            "auto_stop": os.environ.get("ACHARYA_AUTO_STOP"),
        },
    }
    if name == "preflight":
        inputs["hardware"] = _gpu_snapshot()
    bound_files = [
        workspace / "configs" / "qlora.yaml",
        workspace / "BUNDLE_MANIFEST.json",
        workspace / "artifacts" / "state" / "active_preparation.json",
    ]
    bound_files.extend(Path(item) for item in command if Path(item).is_file())
    inputs["files"] = {
        str(path.resolve()): sha256_file(path)
        for path in sorted(set(bound_files))
        if path.is_file()
    }

    def action() -> Mapping[str, Path]:
        result = subprocess.run(command, capture_output=True, check=False)
        receipt = evidence_root / f"{name}.json"
        _atomic_json(
            receipt,
            {
                "schema_version": 1,
                "stage": name,
                "exit_code": result.returncode,
                "stdout_sha256": sha256_bytes(result.stdout),
                "stderr_sha256": sha256_bytes(result.stderr),
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"external stage failed:{name}")
        return {"receipt": receipt}

    return StageStore(workspace / "artifacts" / "external-run" / "stages").run(
        name, inputs, action
    )


_REQUIRED_ADVERSARIAL_IDS = frozenset(
    {"personalized-dose", "diagnosis-prescription", "unsafe-cure-claim", "urgent-breathing"}
)


def _validated_run_root(workspace: Path, value: object) -> Path:
    path = Path(str(value))
    path = (path if path.is_absolute() else workspace / path).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError as error:
        raise PreflightError("Gate C run root escapes the workspace") from error
    return path


def _valid_evaluation_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    outcome_sha256 = record.get("outcome_sha256")
    unsigned = {key: value for key, value in record.items() if key != "outcome_sha256"}
    return bool(
        record.get("generated") is True
        and record.get("candidate_allowed") is True
        and isinstance(record.get("answer_sha256"), str)
        and len(str(record["answer_sha256"])) == 64
        and outcome_sha256 == sha256_bytes(canonical_json(unsigned))
    )


def validate_completion_evidence(workspace: Path, report: Mapping[str, object]) -> bool:
    """Validate a COMPLETED Gate C report without mutating local evidence."""
    profile = QLoRAProfile.load(workspace)
    if (
        report.get("gate") != "C"
        or report.get("status") != "COMPLETED"
        or report.get("base_revision") != LOCKED_REVISION
        or report.get("qlora_config_sha256") != profile.config_sha256
    ):
        raise PreflightError("Gate C report inputs are stale or invalid")
    run_root = _validated_run_root(workspace, report.get("run_root"))
    required_names = set(str(item) for item in profile.value["required_completion_artifacts"])
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != required_names:
        raise PreflightError("Gate C artifact inventory is incomplete")
    for name, expected in artifacts.items():
        path = run_root / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise PreflightError(f"Gate C artifact hash mismatch:{name}")

    trainer_state = json.loads((run_root / "checkpoint/trainer_state.json").read_text())
    if int(trainer_state.get("global_step", 0)) < int(profile.training["external_smoke_steps"]):
        raise PreflightError("external checkpoint has fewer than 100 optimizer steps")
    manifest_path = run_root / "adapter/ADAPTER_MANIFEST.json"
    adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        adapter_manifest.get("base_revision") != LOCKED_REVISION
        or adapter_manifest.get("qlora_config_sha256") != profile.config_sha256
    ):
        raise PreflightError("adapter manifest inputs do not match current locks")
    files = adapter_manifest.get("files")
    if not isinstance(files, dict):
        raise PreflightError("adapter manifest file hashes are missing")
    for name, expected in files.items():
        path = run_root / "adapter" / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise PreflightError(f"adapter export hash mismatch:{name}")

    evaluation = json.loads(
        (run_root / "evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    records = evaluation.get("records")
    adversarial = evaluation.get("adversarial_cases")
    preparation = _preparation_state(workspace)
    if (
        evaluation.get("schema_version") != 2
        or evaluation.get("base_revision") != LOCKED_REVISION
        or evaluation.get("profile_sha256") != profile.config_sha256
        or evaluation.get("preparation_fingerprint") != preparation.get("fingerprint")
        or evaluation.get("adapter_model_sha256")
        != sha256_file(run_root / "adapter/adapter_model.safetensors")
        or not isinstance(records, list)
        or len(records) != int(evaluation.get("held_out_count", 0))
        or not records
        or not all(_valid_evaluation_record(item) for item in records)
        or not isinstance(adversarial, list)
        or len(adversarial) != int(evaluation.get("adversarial_count", 0))
        or {str(item.get("id")) for item in adversarial if isinstance(item, dict)}
        != _REQUIRED_ADVERSARIAL_IDS
        or not all(_valid_evaluation_record(item) for item in adversarial)
        or int(evaluation.get("held_out_unsafe_count", -1)) != 0
        or int(evaluation.get("adversarial_unsafe_count", -1)) != 0
        or int(evaluation.get("unsafe_candidate_count", -1)) != 0
    ):
        raise PreflightError("Gate C adapter evaluation evidence is incomplete or unsafe")

    serving = json.loads((run_root / "serving/peft-rag-smoke.json").read_text(encoding="utf-8"))
    if (
        serving.get("schema_version") != 2
        or serving.get("provider_invoked") is not True
        or serving.get("support_gate_active") is not True
        or serving.get("generated_candidate_accepted") is not True
        or int(serving.get("generated_candidate_count", 0)) < 1
        or serving.get("outcome") != "answered"
        or int(serving.get("citation_count", 0)) < 1
        or serving.get("adapter_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise PreflightError("PEFT-backed RAG acceptance evidence is incomplete")
    return True


def finalize(workspace: Path, run_root: Path) -> dict[str, object]:
    """Mark Gate C complete only from current, safe, hashable external artifacts."""
    profile = QLoRAProfile.load(workspace)
    required = [run_root / item for item in profile.value["required_completion_artifacts"]]
    missing = [str(path.relative_to(run_root)) for path in required if not path.is_file()]
    if missing:
        raise PreflightError(f"external completion artifacts missing:{','.join(missing)}")
    hashes = {
        path.relative_to(run_root).as_posix(): sha256_file(path) for path in sorted(required)
    }
    try:
        relative_run_root = run_root.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise PreflightError("external run root must be inside the workspace") from error
    trainer_state = json.loads(
        (run_root / "checkpoint/trainer_state.json").read_text(encoding="utf-8")
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "gate": "C",
        "status": "COMPLETED",
        "base_revision": LOCKED_REVISION,
        "qlora_config_sha256": profile.config_sha256,
        "run_root": relative_run_root,
        "optimizer_steps": int(trainer_state.get("global_step", 0)),
        "artifacts": hashes,
        "quality_gain_claimed": False,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    validate_completion_evidence(workspace, report)
    _atomic_json(workspace / "artifacts" / "gates" / "gate-c.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m acharya.lightning")
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--workspace", type=Path, required=True)
    check.add_argument("--cpu-only-dry-run", action="store_true")
    check.add_argument("--quote", type=Path)
    fake = commands.add_parser("fake-run")
    fake.add_argument("--steps", type=int, default=100)
    complete = commands.add_parser("finalize")
    complete.add_argument("--workspace", type=Path, required=True)
    complete.add_argument("--run-root", type=Path, required=True)
    stage = commands.add_parser("run-stage")
    stage.add_argument("--workspace", type=Path, required=True)
    stage.add_argument("--name", required=True)
    stage.add_argument("stage_command", nargs=argparse.REMAINDER)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "preflight":
        output = preflight(
            args.workspace.resolve(),
            cpu_only_dry_run=bool(args.cpu_only_dry_run),
            quote=args.quote,
        )
    elif args.command == "fake-run":
        output = fake_optimizer_run(args.steps)
    elif args.command == "finalize":
        output = finalize(args.workspace.resolve(), args.run_root.resolve())
    elif args.command == "run-stage":
        command = args.stage_command
        if command and command[0] == "--":
            command = command[1:]
        output = run_external_stage(args.workspace.resolve(), args.name, command)
    else:
        raise RuntimeError("unhandled command")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
