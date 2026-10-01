from __future__ import annotations

import json
import time

import pytest

from acharya.providers.base import ProviderConfigurationError, ProviderRequest
from acharya.providers.ollama import OllamaProvider
from acharya.rag.prompt import RenderedPrompt


def test_ollama_is_loopback_only_and_preserves_prompt() -> None:
    with pytest.raises(ProviderConfigurationError):
        OllamaProvider(model="m", base_url="https://example.com")
    values: list[bytes] = []

    def transport(request: object, _timeout: float) -> bytes:
        values.append(request.data)  # type: ignore[attr-defined]
        return b'{"response":"grounded"}'

    provider = OllamaProvider(model="m", transport=transport)
    prompt = RenderedPrompt("exact", b"exact", 5, 8, ())
    result = provider.generate(ProviderRequest(prompt, time.monotonic() + 5))
    assert result.text == "grounded"
    assert json.loads(values[0])["prompt"] == "exact"
