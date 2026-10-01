"""Derive independent final gate statuses strictly from local execution artifacts."""

from __future__ import annotations

import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

from acharya.bundle import validate_archive
from acharya.config import Settings, canonical_json, sha256_bytes, sha256_file
from acharya.lightning import (
    PreflightError,
    _preparation_state,
    validate_completion_evidence,
)
from acharya.rag.evaluate import load_calibration
from acharya.rag.index import load_index


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _full_retrieval_ready(workspace: Path, preparation: dict[str, Any] | None) -> bool:
    if preparation is None:
        return False
    try:
        settings = Settings.load(workspace)
        index = load_index(settings, expected_mode="full")
        calibration = load_calibration(settings, index)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False
    return bool(
        index.corpus_fingerprint == preparation.get("fingerprint")
        and index.model_hashes
        and calibration.ready
        and calibration.false_support == 0
    )


def _bundle_evidence(
    workspace: Path, preparation: dict[str, Any] | None
) -> tuple[list[dict[str, object]], bool]:
    expected_fingerprint = preparation.get("fingerprint") if preparation else None
    bundles: list[dict[str, object]] = []
    valid_hashes: list[str] = []
    for path in sorted((workspace / "dist" / "lightning").glob("*.tar.zst")):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        record: dict[str, object] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": digest,
            "validated": False,
        }
        try:
            validation = validate_archive(path)
            matches_preparation = bool(
                expected_fingerprint
                and validation.get("valid") is True
                and validation.get("fingerprint") == expected_fingerprint
            )
            record["validated"] = matches_preparation
            if matches_preparation:
                valid_hashes.append(digest)
        except (OSError, PreflightError, ValueError, tarfile.TarError, zstandard.ZstdError):
            pass
        bundles.append(record)
    ready = len(valid_hashes) >= 2 and len(set(valid_hashes)) == 1
    return bundles, ready


def _gate_a_evidence_valid(workspace: Path, report: dict[str, Any]) -> bool:
    """Validate current Gate A inputs and persisted output artifacts without mutation."""
    try:
        settings = Settings.load(workspace)
        inputs = report["input_hashes"]
        outputs = report["output_hashes"]
        checks = report["checks"]
        calibration = report["calibration_evidence"]
        expected_inputs = {
            "pdf": sha256_file(settings.pdf_path),
            "ingestion_config": settings.config_hash("ingestion"),
            "rag_config": settings.config_hash("rag"),
            "models_config": settings.config_hash("models"),
            "safety_config": settings.config_hash("safety"),
            "evaluation": sha256_file(workspace / "eval" / "golden.jsonl"),
        }
        if (
            report.get("gate") != "A"
            or report.get("status") != "PASSED"
            or inputs != expected_inputs
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
            or not isinstance(calibration, dict)
            or outputs.get("calibration") != sha256_bytes(canonical_json(calibration))
        ):
            return False
        corpus_fingerprint = str(outputs["corpus"])
        corpus_root = workspace / "data" / "processed" / corpus_fingerprint
        corpus = json.loads((corpus_root / "corpus.json").read_text(encoding="utf-8"))
        corpus_payload = {key: value for key, value in corpus.items() if key != "fingerprint"}
        manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
        if (
            corpus.get("fingerprint") != corpus_fingerprint
            or sha256_bytes(canonical_json(corpus_payload)) != corpus_fingerprint
            or manifest.get("fingerprint") != corpus_fingerprint
            or manifest.get("corpus_sha256") != sha256_file(corpus_root / "corpus.json")
            or manifest.get("jsonl_sha256") != sha256_file(corpus_root / "corpus.jsonl")
        ):
            return False
        index_fingerprint = str(outputs["index"])
        index_root = workspace / "artifacts" / "indexes" / index_fingerprint
        index = json.loads((index_root / "index.json").read_text(encoding="utf-8"))
        index_payload = {key: value for key, value in index.items() if key != "fingerprint"}
        if (
            index.get("fingerprint") != index_fingerprint
            or sha256_bytes(canonical_json(index_payload)) != index_fingerprint
            or index.get("mode") != "bm25_only"
            or index.get("corpus_fingerprint") != corpus_fingerprint
            or index.get("config_hash") != settings.config_hash("rag", "models")
            or calibration.get("index_fingerprint") != index_fingerprint
            or calibration.get("corpus_fingerprint") != corpus_fingerprint
            or calibration.get("evaluation_source_sha256") != expected_inputs["evaluation"]
            or calibration.get("ready") is not True
            or calibration.get("false_support") != 0
        ):
            return False
        return all(isinstance(value, str) and len(value) == 64 for value in outputs.values())
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def _gate_c_evidence_valid(workspace: Path, report: dict[str, Any]) -> bool:
    if report.get("status") != "COMPLETED":
        return False
    try:
        return validate_completion_evidence(workspace, report)
    except (KeyError, OSError, TypeError, ValueError, PreflightError):
        return False


def derive(workspace: Path) -> dict[str, object]:
    gates = workspace / "artifacts" / "gates"
    gate_a = _read(gates / "gate-a.json") or {"gate": "A", "status": "PENDING"}
    gate_a_valid = _gate_a_evidence_valid(workspace, gate_a)
    existing_b = _read(gates / "gate-b.json") or {"gate": "B", "status": "PENDING"}
    gate_c = _read(gates / "gate-c.json") or {
        "gate": "C",
        "status": "PENDING_EXTERNAL_GPU",
        "reason": "no real external GPU artifacts",
    }
    provider = _read(workspace / "artifacts" / "provider-check" / "kiro.json")
    try:
        preparation_state = _preparation_state(workspace)
        preparation = preparation_state if preparation_state.get("ready") is True else None
    except PreflightError:
        preparation = None
    bundles, bundle_ready = _bundle_evidence(workspace, preparation)
    blocked_data = existing_b.get("status") == "BLOCKED_EXTERNAL_DATA"
    full_retrieval_ready = _full_retrieval_ready(workspace, preparation)
    gate_b_status = "BLOCKED_EXTERNAL_DATA" if blocked_data else "PENDING"
    if preparation is not None and bundle_ready and full_retrieval_ready and not blocked_data:
        gate_b_status = "PASSED"
    existing_components: dict[str, Any] = {}
    if isinstance(existing_b.get("components"), dict):
        existing_components = existing_b["components"]
    gate_b: dict[str, object] = {
        **existing_b,
        "schema_version": 1,
        "gate": "B",
        "status": gate_b_status,
        "components": {
            **existing_components,
            "provider": provider
            or {"status": "UNVERIFIED", "category": "provider_check_missing"},
            "ios_macos": {
                "status": "PENDING_MACOS_VALIDATION",
                "reason": "xcodebuild is unavailable on this Linux host",
            },
            "transfer_bundle": {
                "status": (
                    "PASSED"
                    if bundle_ready
                    else (
                        "BLOCKED_BUNDLE_VALIDATION"
                        if preparation is not None
                        else "BLOCKED_STRICT_PREPARATION"
                    )
                ),
                "bundles": bundles,
            },
            "full_retrieval": {
                "status": "PASSED" if full_retrieval_ready else "PENDING",
            },
        },
    }
    if not _gate_c_evidence_valid(workspace, gate_c):
        gate_c = {
            "schema_version": 1,
            "gate": "C",
            "status": "PENDING_EXTERNAL_GPU",
            "reason": "no current safe hashed external checkpoint, evaluation, and serving run",
            "quality_gain_claimed": False,
        }
    gate_a_path = gates / "gate-a.json"
    summary: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "gate_a": {
            "status": "PASSED" if gate_a_valid else "PENDING",
            "report_sha256": (
                sha256_file(gate_a_path) if gate_a_valid and gate_a_path.is_file() else None
            ),
        },
        "gate_b": {
            "status": gate_b_status,
            "external_limitations": [
                "macOS validation",
                "provider exact-response check",
            ],
        },
        "gate_c": {"status": gate_c.get("status"), "quality_gain_claimed": False},
    }
    gates.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (gates / "gate-b.json", gate_b),
        (gates / "gate-c.json", gate_c),
        (gates / "final-summary.json", summary),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(canonical_json(value) + b"\n")
        os.replace(temporary, path)
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.workspace.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
