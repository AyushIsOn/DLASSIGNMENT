from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from acharya.providers.base import ProviderConfigurationError, ProviderContext, ProviderRequest
from acharya.providers.peft_local import PEFTLocalProvider
from acharya.rag.prompt import RenderedPrompt


def test_peft_requires_hashes_and_rag_context(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weight = adapter / "adapter.bin"
    weight.write_bytes(b"weight")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_revision": "revision",
                "files": {"adapter.bin": hashlib.sha256(b"weight").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    provider = PEFTLocalProvider(
        adapter_path=adapter.resolve(),
        manifest_path=manifest.resolve(),
        base_revision="revision",
        generator=lambda prompt, _tokens: prompt,
    )
    empty = RenderedPrompt("p", b"p", 1, 8, ())
    with pytest.raises(ProviderConfigurationError, match="RAG"):
        provider.generate(ProviderRequest(empty, time.monotonic() + 5))
    context = ProviderContext("a", "s", 1, "q", "text", "educational")
    grounded = RenderedPrompt("p", b"p", 1, 8, (context,))
    assert provider.generate(ProviderRequest(grounded, time.monotonic() + 5)).text == "p"
    weight.write_bytes(b"changed")
    with pytest.raises(ProviderConfigurationError, match="hash"):
        PEFTLocalProvider(
            adapter_path=adapter.resolve(),
            manifest_path=manifest.resolve(),
            base_revision="revision",
        )
