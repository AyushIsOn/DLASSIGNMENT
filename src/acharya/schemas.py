"""Public API schemas (the iOS app decodes ChatResponse and the error envelope)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChatMessage(_Frozen):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class ChatRequest(_Frozen):
    message: str = Field(min_length=1, max_length=4000)
    history: tuple[ChatMessage, ...] = Field(default=(), max_length=20)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("history")
    @classmethod
    def complete_pairs(cls, value: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        if len(value) % 2:
            raise ValueError("history must contain complete user/assistant pairs")
        for index, message in enumerate(value):
            if message.role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("history must alternate user and assistant roles")
        return value


class Citation(_Frozen):
    chunk_id: str
    source: str
    page: int = 0
    text: str


class RetrievalScores(_Frozen):
    bm25: float
    dense: float | None = None
    rerank: float | None = None


class ChatResponse(_Frozen):
    answer: str
    citations: tuple[Citation, ...] = ()
    scores: tuple[RetrievalScores, ...] = ()
    mode: str  # "finetuned_rag" | "finetuned" | "safety"
    warning: str | None = None
    outcome: Literal["answered", "urgent", "refused"]
    model: str | None = None
    latency_ms: int | None = None


class HealthResponse(_Frozen):
    status: Literal["live", "ready", "not_ready"]
    ready: bool
    model: str
    backend: str
    knowledge_base_entries: int
    detail: str | None = None
