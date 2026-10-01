"""Gate B acquisition and preparation orchestration."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import cast

from acharya.config import Settings, canonical_json
from acharya.ingest.download import AcquisitionError, acquire_dataset, blocker_record
from training.prepare_sft import (  # type: ignore[import-untyped]
    Preparation,
    prepare,
    verify_preparation,
)


def _write_gate(settings: Settings, value: dict[str, object]) -> None:
    path = settings.workspace / "artifacts" / "gates" / "gate-b.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def prepare_handoff(settings: Settings, *, strict: bool) -> Preparation:
    acquisitions = []
    configured = settings.datasets["kaggle"]
    blockers: list[tuple[AcquisitionError, dict[str, object]]] = []
    for dataset_id, spec in configured["datasets"].items():
        try:
            acquisitions.append(
                acquire_dataset(
                    settings.workspace, dataset_id, spec, configured["limits"], strict=strict
                )
            )
        except AcquisitionError as error:
            blockers.append((error, blocker_record(error)))
    if blockers:
        _write_gate(
            settings,
            {
                "schema_version": 1,
                "gate": "B",
                "status": "BLOCKED_EXTERNAL_DATA",
                "component": "dataset_acquisition",
                "evidence": {"blockers": [record for _, record in blockers]},
            },
        )
        raise blockers[0][0]
    try:
        result = prepare(settings, tuple(acquisitions))
    except BaseException as error:
        _write_gate(
            settings,
            {
                "schema_version": 1,
                "gate": "B",
                "status": "BLOCKED_EXTERNAL_DATA",
                "component": "dataset_preparation",
                "evidence": {
                    "failure_category": type(error).__name__,
                    "message": str(error)[:300],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "redacted": True,
                },
            },
        )
        raise
    _write_gate(
        settings,
        {
            "schema_version": 1,
            "gate": "B",
            "status": "DATA_PREPARED",
            "component": "dataset_preparation",
            "fingerprint": result.fingerprint,
            "retrieval_count": result.retrieval_count,
            "split_counts": result.split_counts,
            "source_counts": result.source_counts,
            "hashes": result.hashes,
        },
    )
    return result


def verify_handoff(settings: Settings, *, strict: bool) -> dict[str, object]:
    gate_path = settings.workspace / "artifacts" / "gates" / "gate-b.json"
    if strict and not gate_path.is_file():
        raise RuntimeError("Gate B evidence is missing")
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "BLOCKED_EXTERNAL_DATA":
            raise RuntimeError("Gate B is blocked by external data")
    return cast(dict[str, object], verify_preparation(settings, strict=strict))
