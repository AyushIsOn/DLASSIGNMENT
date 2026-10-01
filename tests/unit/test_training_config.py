from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acharya.lightning import LOCKED_REVISION, PreflightError, QLoRAProfile, cpu_dry_run


def test_locked_qlora_profile_and_cpu_dry_run(project_root: Path) -> None:
    profile = QLoRAProfile.load(project_root)
    assert profile.value["base_model"]["revision"] == LOCKED_REVISION
    assert profile.value["quantization"] == {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }
    assert profile.value["training"]["effective_batch_size"] == 16
    result = cpu_dry_run()
    assert result["steps"] == 5
    assert result["saved_checkpoint"] is False
    assert result["losses"] == sorted(result["losses"], reverse=True)


def test_profile_rejects_full_tuning_or_relaxed_caps(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    path = root / "configs" / "qlora.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["quantization"]["load_in_4bit"] = False
    value["external"]["maximum_credits"] = 20
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(PreflightError):
        QLoRAProfile.load(root)
