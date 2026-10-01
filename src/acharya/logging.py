"""Structured logging with deterministic secret redaction."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token|credential)", re.I)
_SECRET_VALUE = re.compile(r"(?i)(?:ksk_|sk-)[A-Za-z0-9_-]{8,}|Bearer\s+\S+")
REDACTED = "[REDACTED]"


def redact(value: Any, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return REDACTED
    if isinstance(value, str):
        return _SECRET_VALUE.sub(REDACTED, value)
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload["fields"] = redact(fields)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
