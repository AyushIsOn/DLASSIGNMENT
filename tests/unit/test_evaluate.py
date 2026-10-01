from __future__ import annotations

import json
from pathlib import Path

import pytest

from acharya.config import Settings
from acharya.rag.evaluate import calibrate, is_ready, load_calibration
from acharya.rag.index import load_index


def test_calibration_has_zero_false_support_and_matching_hashes(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    result = load_calibration(settings)
    assert result.ready
    assert result.false_support == 0
    assert result.thresholds.raw_score != 0.75


def test_missing_and_stale_calibration_refuse_readiness(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    path = settings.state_dir / "bm25_calibration.json"
    value = json.loads(path.read_text())
    value["index_fingerprint"] = "stale"
    path.write_text(json.dumps(value))
    assert not is_ready(settings)
    with pytest.raises(RuntimeError, match="stale"):
        load_calibration(settings)
    calibrate(settings, load_index(settings))
    assert is_ready(settings)
    evaluation = settings.workspace / "eval" / "golden.jsonl"
    evaluation.write_text(evaluation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not is_ready(settings)
    with pytest.raises(RuntimeError, match="stale"):
        load_calibration(settings)
