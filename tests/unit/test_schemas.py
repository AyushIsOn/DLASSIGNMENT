from __future__ import annotations

import pytest
from pydantic import ValidationError

from acharya.schemas import ChatRequest, GateState, GateStatus


def test_models_are_immutable_and_strict() -> None:
    status = GateStatus(gate="A", status=GateState.PASSED)
    with pytest.raises(ValidationError):
        status.status = GateState.FAILED
    with pytest.raises(ValidationError):
        GateStatus(gate="A", status=GateState.PASSED, unknown=True)


def test_chat_rejects_blank_excess_and_malformed_history() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")
    history = [{"role": "user", "content": "hello"}] * 21
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", history=history)
    malformed = (
        [{"role": "user", "content": "odd"}],
        [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "second"},
        ],
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ],
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
            {"role": "user", "content": "four"},
        ],
    )
    for value in malformed:
        with pytest.raises(ValidationError):
            ChatRequest(message="hello", history=value)
