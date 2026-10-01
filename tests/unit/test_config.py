from __future__ import annotations

from pathlib import Path

import pytest

from acharya.config import ConfigurationError, Settings, canonical_json, sha256_bytes


def test_settings_loads_without_dotenv(project_root: Path) -> None:
    settings = Settings.load(project_root)
    assert settings.pdf_path.name == "dataset V1.0.pdf"
    assert len(settings.config_hash("rag")) == 64


def test_env_template_has_blank_context_window(project_root: Path) -> None:
    lines = (project_root / ".env.example").read_text().splitlines()
    assert "KIRO_CONTEXT_WINDOW_TOKENS=" in lines


def test_canonical_hash_is_order_independent() -> None:
    assert sha256_bytes(canonical_json({"b": 2, "a": 1})) == sha256_bytes(
        canonical_json({"a": 1, "b": 2})
    )


def test_invalid_workspace_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        Settings.load(tmp_path)
