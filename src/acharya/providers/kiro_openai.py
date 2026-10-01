"""OpenAI-compatible Kiro HTTP provider with pre-DNS validation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

from acharya.providers.base import (
    ProviderConfigurationError,
    ProviderOutputError,
    ProviderRequest,
    ProviderResult,
    ProviderTimeoutError,
    ProviderTransportError,
)

Transport = Callable[[urllib.request.Request, float], bytes]


def _safe_base_url(value: str, *, loopback_http: bool = False) -> str:
    parsed = urlsplit(value)
    loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme != "https"
            and not (loopback_http and loopback and parsed.scheme == "http")
        )
    ):
        raise ProviderConfigurationError("provider URL is unsafe")
    return value.rstrip("/")


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read(1_048_577))


class KiroOpenAIProvider:
    name = "kiro-openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        context_window_tokens: int,
        transport: Transport = _urlopen,
    ) -> None:
        if not api_key.strip() or not model.strip() or context_window_tokens <= 0:
            raise ProviderConfigurationError("incomplete Kiro HTTP configuration")
        self.base_url = _safe_base_url(base_url, loopback_http=True)
        self.api_key = api_key
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
                "messages": [{"role": "user", "content": request.prompt.text}],
                "max_tokens": request.prompt.output_tokens,
                "temperature": 0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
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
            value = json.loads(raw)
            text = value["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
            raise ProviderOutputError() from error
        if not text:
            raise ProviderOutputError()
        return ProviderResult(text, (), self.name)
