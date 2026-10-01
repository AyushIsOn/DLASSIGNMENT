from __future__ import annotations

import json
import logging
from pathlib import Path

from acharya.cli import _provider_check
from acharya.logging import REDACTED, JsonFormatter, redact
from acharya.providers.base import ProviderResult


def test_redacts_secret_keys_and_values() -> None:
    value = redact({"api_key": "sentinel", "note": "Bearer abc", "nested": ["sk-123456789"]})
    assert value == {"api_key": REDACTED, "note": REDACTED, "nested": [REDACTED]}


def test_formatter_emits_redacted_json() -> None:
    record = logging.LogRecord("test", logging.INFO, "", 1, "key ksk_abcdefghijk", (), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "key [REDACTED]"


def test_provider_health_record_is_allowlisted(tmp_path: Path, monkeypatch: object) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("KIRO_API_KEY=ksk_testsentinelvalue\n", encoding="utf-8")
    record_path = tmp_path / "kiro.json"

    class FakeKiro:
        version = "2.26.0"

        def __init__(self, *, environment: dict[str, str]) -> None:
            assert environment["KIRO_API_KEY"].startswith("ksk_")

        def generate(self, request: object) -> ProviderResult:
            return ProviderResult("KIRO_API_OK", (), "kiro-cli")

    monkeypatch.setattr("acharya.cli.KiroCLIProvider", FakeKiro)  # type: ignore[attr-defined]
    record, code = _provider_check(env_file, record_path)
    assert code == 0 and record["status"] == "PASSED"
    raw = record_path.read_text(encoding="utf-8")
    assert "testsentinel" not in raw
    assert set(json.loads(raw)) == {
        "schema_version",
        "provider",
        "cli_version",
        "prompt_id",
        "status",
        "category",
        "exit_code",
        "checked_at",
        "credential_recorded",
    }
