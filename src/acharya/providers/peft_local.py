"""Hash-locked optional local PEFT provider."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from acharya.config import sha256_file
from acharya.providers.base import (
    ProviderConfigurationError,
    ProviderOutputError,
    ProviderRequest,
    ProviderResult,
    ProviderTimeoutError,
)

Generator = Callable[[str, int], str]


class PEFTLocalProvider:
    name = "peft-local"

    def __init__(
        self,
        *,
        adapter_path: Path,
        manifest_path: Path,
        base_revision: str,
        generator: Generator | None = None,
    ) -> None:
        adapter = adapter_path.expanduser()
        manifest_file = manifest_path.expanduser()
        if not adapter.is_absolute() or not manifest_file.is_absolute():
            raise ProviderConfigurationError("PEFT paths must be absolute")
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise ProviderConfigurationError("PEFT manifest is unavailable") from error
        if manifest.get("base_revision") != base_revision or not base_revision:
            raise ProviderConfigurationError("PEFT base revision mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ProviderConfigurationError("PEFT file hashes are missing")
        for relative, expected in sorted(files.items()):
            path = (adapter / str(relative)).resolve()
            try:
                path.relative_to(adapter.resolve())
            except ValueError as error:
                raise ProviderConfigurationError("PEFT manifest path escapes adapter") from error
            if not path.is_file() or sha256_file(path) != expected:
                raise ProviderConfigurationError("PEFT adapter hash mismatch")
        self.adapter_path = adapter.resolve()
        self.base_revision = base_revision
        self._generator = generator

    def generate(self, request: ProviderRequest) -> ProviderResult:
        if not request.prompt.contexts:
            raise ProviderConfigurationError("PEFT generation cannot disable RAG")
        if request.deadline_monotonic <= time.monotonic():
            raise ProviderTimeoutError()
        if self._generator is None:
            raise ProviderConfigurationError("PEFT runtime is not configured")
        text = self._generator(request.prompt.text, request.prompt.output_tokens).strip()
        if not text:
            raise ProviderOutputError()
        return ProviderResult(text, (), self.name)
