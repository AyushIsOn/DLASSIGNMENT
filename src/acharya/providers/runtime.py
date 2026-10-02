"""Explicit opt-in provider wiring for the API; extraction needs no model runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from acharya.providers.base import GenerativeProvider, ProviderConfigurationError
from acharya.providers.ollama import OllamaProvider
from acharya.providers.peft_local import PEFTLocalProvider


def configured_provider(
    workspace: Path, environment: Mapping[str, str]
) -> GenerativeProvider | None:
    name = environment.get("ACHARYA_PROVIDER", "extractive")
    if name == "extractive":
        return None
    if name == "ollama":
        return OllamaProvider(
            model=environment.get("ACHARYA_MODEL", ""),
            base_url=environment.get("ACHARYA_OLLAMA_URL", "http://127.0.0.1:11434"),
        )
    if name == "peft-local":
        from acharya.lightning import LOCKED_REVISION

        value = environment.get("ACHARYA_ADAPTER_PATH", "")
        if not value:
            raise ProviderConfigurationError("ACHARYA_ADAPTER_PATH is required")
        adapter = Path(value).expanduser().resolve()
        provider = PEFTLocalProvider(
            adapter_path=adapter,
            manifest_path=adapter / "ADAPTER_MANIFEST.json",
            base_revision=LOCKED_REVISION,
        )
        # Validate the manifest before importing or loading the optional GPU runtime.
        from training.runtime import make_generator

        provider.set_generator(make_generator(workspace, adapter))
        return provider
    raise ProviderConfigurationError(f"unknown ACHARYA_PROVIDER: {name}")
