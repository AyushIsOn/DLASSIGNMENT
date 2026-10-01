"""Public API and immutable evidence schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GateState(StrEnum):
    PASSED = "PASSED"
    PENDING = "PENDING"
    PENDING_EXTERNAL_GPU = "PENDING_EXTERNAL_GPU"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class GateStatus(ImmutableModel):
    gate: Literal["A", "B", "C"]
    status: GateState
    evidence: tuple[str, ...] = ()
    reason: str | None = None


class ChatMessage(ImmutableModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class ChatRequest(ImmutableModel):
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
    def history_has_complete_pairs(
        cls, value: tuple[ChatMessage, ...]
    ) -> tuple[ChatMessage, ...]:
        if len(value) % 2:
            raise ValueError("history must contain complete user/assistant pairs")
        for index, message in enumerate(value):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("history must alternate user and assistant roles")
        return value


class Citation(ImmutableModel):
    chunk_id: str
    source: str
    page: int
    text: str


class RetrievalScores(ImmutableModel):
    bm25: float
    dense: float | None = None
    rerank: float | None = None


class ChatResponse(ImmutableModel):
    answer: str
    citations: tuple[Citation, ...]
    scores: tuple[RetrievalScores, ...]
    mode: Literal["bm25_only", "mvp_hybrid", "full"] = "bm25_only"
    warning: Literal["bm25_only_fallback", "mvp_uncalibrated_retrieval"] | None = (
        "bm25_only_fallback"
    )
    outcome: Literal["answered", "abstained", "urgent", "refused"]


class HealthResponse(ImmutableModel):
    status: Literal["live", "ready", "not_ready"]
    retrieval_mode: Literal["bm25_only", "mvp_hybrid", "full"] = "bm25_only"
    ready: bool
    gates: tuple[GateStatus, GateStatus, GateStatus]


class ErrorBody(ImmutableModel):
    code: str
    message: str


class EvidenceReport(ImmutableModel):
    gate: Literal["A"] = "A"
    status: GateState
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]
    checks: dict[str, bool]
