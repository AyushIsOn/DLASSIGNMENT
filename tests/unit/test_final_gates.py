from __future__ import annotations

import json
from pathlib import Path

import pytest

from acharya import final_gates


def test_bundle_evidence_requires_valid_matching_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "dist" / "lightning"
    bundle_dir.mkdir(parents=True)
    for name in ("one.tar.zst", "two.tar.zst"):
        (bundle_dir / name).write_bytes(b"not-an-archive")

    evidence, ready = final_gates._bundle_evidence(
        tmp_path, {"fingerprint": "a" * 64}
    )
    assert ready is False
    assert all(record["validated"] is False for record in evidence)

    monkeypatch.setattr(
        final_gates,
        "validate_archive",
        lambda _path: {"valid": True, "fingerprint": "a" * 64},
    )
    evidence, ready = final_gates._bundle_evidence(
        tmp_path, {"fingerprint": "a" * 64}
    )
    assert ready is True
    assert all(record["validated"] is True for record in evidence)

    _, ready = final_gates._bundle_evidence(tmp_path, {"fingerprint": "b" * 64})
    assert ready is False


def test_forged_gate_reports_fail_closed(tmp_path: Path) -> None:
    gates = tmp_path / "artifacts" / "gates"
    gates.mkdir(parents=True)
    (gates / "gate-a.json").write_text(
        json.dumps({"gate": "A", "status": "PASSED", "checks": {"forged": True}}),
        encoding="utf-8",
    )
    (gates / "gate-c.json").write_text(
        json.dumps({"gate": "C", "status": "COMPLETED", "artifacts": {}}),
        encoding="utf-8",
    )
    summary = final_gates.derive(tmp_path)
    assert summary["gate_a"]["status"] == "PENDING"  # type: ignore[index]
    assert summary["gate_c"]["status"] == "PENDING_EXTERNAL_GPU"  # type: ignore[index]
    rewritten = json.loads((gates / "gate-c.json").read_text(encoding="utf-8"))
    assert rewritten["status"] == "PENDING_EXTERNAL_GPU"
