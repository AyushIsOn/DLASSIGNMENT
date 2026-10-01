from __future__ import annotations

import json
import time

import pytest

from acharya.providers.base import ProviderConfigurationError, ProviderRequest
from acharya.providers.kiro_openai import KiroOpenAIProvider
from acharya.rag.prompt import RenderedPrompt


def _request() -> ProviderRequest:
    prompt = RenderedPrompt("prompt", b"prompt", 6, 12, ())
    return ProviderRequest(prompt, time.monotonic() + 5)


def test_http_validates_before_transport_and_uses_exact_prompt() -> None:
    calls: list[bytes] = []

    def transport(request: object, _timeout: float) -> bytes:
        calls.append(request.data)  # type: ignore[attr-defined]
        return b'{"choices":[{"message":{"content":"answer"}}]}'

    with pytest.raises(ProviderConfigurationError):
        KiroOpenAIProvider(
            api_key="key",
            base_url="http://external.example",
            model="model",
            context_window_tokens=100,
            transport=transport,
        )
    assert calls == []
    provider = KiroOpenAIProvider(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="model",
        context_window_tokens=100,
        transport=transport,
    )
    assert provider.generate(_request()).text == "answer"
    assert json.loads(calls[0])["messages"][0]["content"] == "prompt"
