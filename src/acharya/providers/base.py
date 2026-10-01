"""Immutable provider contracts and normalized failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from acharya.rag.prompt import RenderedPrompt


@dataclass(frozen=True)
class ProviderContext:
    chunk_id: str
    source: str
    page: int
    question: str
    text: str
    role: str


@dataclass(frozen=True)
class ProviderRequest:
    prompt: RenderedPrompt
    deadline_monotonic: float


@dataclass(frozen=True)
class ProviderResult:
    text: str
    used_chunk_ids: tuple[str, ...]
    provider: str


class GenerativeProvider(Protocol):
    name: str

    def generate(self, request: ProviderRequest) -> ProviderResult: ...


class ProviderError(RuntimeError):
    """Normalized provider failure safe to categorize without raw transport details."""

    category = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str = "provider request failed",
        *,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ProviderConfigurationError(ProviderError):
    category = "invalid_configuration"


class ProviderTimeoutError(ProviderError):
    category = "timeout"
    retryable = True


class ProviderTransportError(ProviderError):
    category = "transport"
    retryable = True


class ProviderOutputError(ProviderError):
    category = "invalid_output"
