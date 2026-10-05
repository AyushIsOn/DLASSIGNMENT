from __future__ import annotations

import json
from pathlib import Path

import pytest

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.lightning import (
    PreflightError,
    QLoRAProfile,
    StageStore,
    fake_optimizer_run,
    finalize,
    preflight,
    select_resume_checkpoint,
    validate_quote,
)


def test_cpu_preflight_and_fake_100_steps(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    result = preflight(root, cpu_only_dry_run=True)
    assert result["status"] == "PENDING_EXTERNAL_GPU"
    assert result["paid_training_started"] is False
    assert result["preparation"]["ready"] is False  # type: ignore[index]
    assert fake_optimizer_run(100)["completed_steps"] == 100
    gate = json.loads((root / "artifacts/gates/gate-c.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PENDING_EXTERNAL_GPU"
    assert gate["checkpoint_claimed"] is False


def test_stage_markers_are_input_bound_and_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    calls = 0

    def action() -> dict[str, Path]:
        nonlocal calls
        calls += 1
        output.write_text(str(calls), encoding="utf-8")
        return {"result": output}

    store = StageStore(tmp_path / "stages")
    first = store.run("profile", {"fingerprint": "one"}, action)
    assert first["status"] == "COMPLETED"
    assert store.run("profile", {"fingerprint": "one"}, action) == first
    assert calls == 1
    store.run("profile", {"fingerprint": "two"}, action)
    assert calls == 2

    def fail() -> dict[str, Path]:
        raise RuntimeError("stopped")

    with pytest.raises(RuntimeError):
        store.run("train", {"fingerprint": "one"}, fail)
    marker = json.loads((tmp_path / "stages/train.json").read_text(encoding="utf-8"))
    assert marker["status"] == "FAILED"


def test_resume_and_quote_caps(workspace_factory: object, tmp_path: Path) -> None:
    root = workspace_factory()  # type: ignore[operator]
    checkpoints = tmp_path / "checkpoints"
    for step, binding in ((25, "run"), (50, "other"), (75, "run")):
        path = checkpoints / f"checkpoint-{step}"
        path.mkdir(parents=True)
        (path / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        (path / "RUN_INPUT.json").write_text(json.dumps({"run_sha256": binding}), encoding="utf-8")
        from acharya.checkpoints import seal

        for filename in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (path / filename).write_text("fixture")
        seal(path)
    assert select_resume_checkpoint(checkpoints, "run") == checkpoints / "checkpoint-75"

    quote = tmp_path / "quote.json"
    quote.write_text(
        json.dumps(
            {
                "credits_per_hour": 2,
                "available_credits": 10,
                "currency_per_credit": 1.5,
                "currency": "USD",
                "quoted_at_utc": "2025-01-01T00:00:00Z",
                "startup_minutes": 10,
                "remaining_training_minutes": 100,
                "evaluation_export_minutes": 30,
            }
        ),
        encoding="utf-8",
    )
    result = validate_quote(QLoRAProfile.load(root), quote)
    assert result["quote_sha256"] == sha256_file(quote)
    quote.write_text(quote.read_text().replace('"credits_per_hour": 2', '"credits_per_hour": 9'))
    with pytest.raises(PreflightError, match="available"):
        validate_quote(QLoRAProfile.load(root), quote)


def _completion_fixture(root: Path) -> Path:
    fingerprint = "a" * 64
    processed = root / "data" / "processed" / fingerprint
    reports = root / "artifacts" / "preparation" / fingerprint
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    for split in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (processed / split).write_text("{}\n", encoding="utf-8")
    (reports / "manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "hashes": {}}), encoding="utf-8"
    )
    state = root / "artifacts" / "state"
    state.mkdir(parents=True)
    (state / "active_preparation.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "processed_path": str(processed),
                "report_path": str(reports),
            }
        ),
        encoding="utf-8",
    )
    run = root / "artifacts" / "external-run"
    for relative, payload in {
        "checkpoint/trainer_state.json": b'{"global_step":100}',
        "adapter/adapter_config.json": b"{}",
        "adapter/adapter_model.safetensors": b"adapter",
        "adapter/tokenizer_config.json": b"{}",
    }.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    profile = QLoRAProfile.load(root)
    adapter_files = {
        name: sha256_file(run / "adapter" / name)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
        )
    }
    (run / "adapter/ADAPTER_MANIFEST.json").write_bytes(
        canonical_json(
            {
                "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "qlora_config_sha256": profile.config_sha256,
                "files": adapter_files,
            }
        )
        + b"\n"
    )
    base_record = {
        "id": "held",
        "generated": True,
        "answer_sha256": "b" * 64,
        "candidate_allowed": True,
    }
    base_record["outcome_sha256"] = sha256_bytes(canonical_json(base_record))
    adversarial = []
    for case_id in (
        "personalized-dose",
        "diagnosis-prescription",
        "unsafe-cure-claim",
        "urgent-breathing",
    ):
        record = {**base_record, "id": case_id}
        record.pop("outcome_sha256")
        record["outcome_sha256"] = sha256_bytes(canonical_json(record))
        adversarial.append(record)
    evaluation = {
        "schema_version": 2,
        "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "profile_sha256": profile.config_sha256,
        "preparation_fingerprint": fingerprint,
        "adapter_model_sha256": sha256_file(run / "adapter/adapter_model.safetensors"),
        "held_out_count": 1,
        "held_out_unsafe_count": 0,
        "unsafe_candidate_count": 0,
        "adversarial_count": 4,
        "adversarial_unsafe_count": 0,
        "records": [base_record],
        "adversarial_cases": adversarial,
    }
    evaluation_path = run / "evaluation/evaluation.json"
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_bytes(canonical_json(evaluation) + b"\n")
    (run / "evaluation/acceptance.json").write_text(
        json.dumps(
            {
                "engineering_acceptance_passed": True,
                "retrieval_mode": "full",
                "corpus_fingerprint": fingerprint,
                "adapter_model_sha256": sha256_file(run / "adapter/adapter_model.safetensors"),
                "test_sha256": sha256_file(root / "eval/acceptance.jsonl"),
            }
        )
    )
    serving_path = run / "serving/peft-rag-smoke.json"
    serving_path.parent.mkdir(parents=True)
    serving_path.write_bytes(
        canonical_json(
            {
                "schema_version": 2,
                "provider_invoked": True,
                "support_gate_active": True,
                "generated_candidate_accepted": True,
                "generated_candidate_count": 1,
                "outcome": "answered",
                "citation_count": 1,
                "adapter_manifest_sha256": sha256_file(run / "adapter/ADAPTER_MANIFEST.json"),
            }
        )
        + b"\n"
    )
    return run


def test_finalize_rejects_unsafe_and_invocation_only_evidence(
    workspace_factory: object,
) -> None:
    root = workspace_factory()  # type: ignore[operator]
    run = _completion_fixture(root)
    evaluation_path = run / "evaluation/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["unsafe_candidate_count"] = 1
    evaluation_path.write_bytes(canonical_json(evaluation) + b"\n")
    with pytest.raises(PreflightError, match="unsafe"):
        finalize(root, run)
    evaluation["unsafe_candidate_count"] = 0
    evaluation_path.write_bytes(canonical_json(evaluation) + b"\n")
    serving_path = run / "serving/peft-rag-smoke.json"
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    serving["generated_candidate_accepted"] = False
    serving_path.write_bytes(canonical_json(serving) + b"\n")
    with pytest.raises(PreflightError, match="acceptance"):
        finalize(root, run)


def test_finalize_rejects_failed_api_acceptance(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    run = _completion_fixture(root)
    path = run / "evaluation/acceptance.json"
    report = json.loads(path.read_text())
    report["engineering_acceptance_passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(PreflightError, match="API/RAG"):
        finalize(root, run)
