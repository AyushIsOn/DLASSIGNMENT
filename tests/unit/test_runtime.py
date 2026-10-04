from pathlib import Path

import pytest

from acharya.providers.base import ProviderConfigurationError
from acharya.providers.ollama import OllamaProvider
from acharya.providers.runtime import configured_provider


def test_provider_selection_is_explicit(tmp_path: Path) -> None:
    assert configured_provider(tmp_path, {}) is None
    assert isinstance(
        configured_provider(
            tmp_path,
            {
                "ACHARYA_PROVIDER": "ollama",
                "ACHARYA_MODEL": "test",
            },
        ),
        OllamaProvider,
    )
    with pytest.raises(ProviderConfigurationError):
        configured_provider(tmp_path, {"ACHARYA_PROVIDER": "peft-local"})
    with pytest.raises(ProviderConfigurationError):
        configured_provider(tmp_path, {"ACHARYA_PROVIDER": "unknown"})
