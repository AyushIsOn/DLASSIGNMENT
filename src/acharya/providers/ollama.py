"""Loopback-only Ollama development provider."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from acharya.providers.base import (
    ProviderConfigurationError,
    ProviderOutputError,
    ProviderRequest,
    ProviderResult,
    ProviderTimeoutError,
    ProviderTransportError,
)
from acharya.providers.kiro_openai import Transport, _safe_base_url, _urlopen


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        context_window_tokens: int = 8192,
        transport: Transport = _urlopen,
    ) -> None:
        if not model.strip() or context_window_tokens <= 0:
            raise ProviderConfigurationError("incomplete Ollama configuration")
        safe = _safe_base_url(base_url, loopback_http=True)
        parsed = urllib.parse.urlsplit(safe)
        if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ProviderConfigurationError("Ollama must use loopback")
        self.base_url = safe
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.transport = transport

    def generate(self, request: ProviderRequest) -> ProviderResult:
        timeout = request.deadline_monotonic - time.monotonic()
        if timeout <= 0:
            raise ProviderTimeoutError()
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": request.prompt.text,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": request.prompt.output_tokens,
                    "num_ctx": self.context_window_tokens,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        http_request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = self.transport(http_request, timeout)
        except TimeoutError as error:
            raise ProviderTimeoutError() from error
        except (OSError, urllib.error.URLError) as error:
            raise ProviderTransportError() from error
        if len(raw) > 1_048_576:
            raise ProviderOutputError()
        try:
            text = str(json.loads(raw)["response"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderOutputError() from error
        if not text:
            raise ProviderOutputError()
        return ProviderResult(text, (), self.name)
