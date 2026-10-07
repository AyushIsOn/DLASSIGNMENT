"""Deterministic medical-safety routing around the model (before and after it)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Action(StrEnum):
    URGENT = "urgent"
    SELF_HARM = "self_harm"
    DOSE = "dose"
    ANSWER = "answer"


@dataclass(frozen=True)
class Decision:
    action: Action
    response: str | None = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return " ".join(text.split())


class SafetyPolicy:
    ORDER = (Action.SELF_HARM, Action.URGENT, Action.DOSE)

    def __init__(self, config: dict[str, Any]) -> None:
        self.responses = {Action(key): str(value) for key, value in config["responses"].items()}
        self.patterns = {
            Action(key): tuple(re.compile(pattern, re.I) for pattern in patterns)
            for key, patterns in config["query_patterns"].items()
        }
        self.dose_in_answer = re.compile(config["dose_in_answer"], re.I)
        for action in self.ORDER:
            if action not in self.responses or action not in self.patterns:
                raise ValueError(f"safety config is missing {action.value}")

    @classmethod
    def load(cls, path: Path) -> SafetyPolicy:
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    def classify(self, message: str) -> Decision:
        text = _normalize(message)
        for action in self.ORDER:
            if any(pattern.search(text) for pattern in self.patterns[action]):
                return Decision(action, self.responses[action])
        return Decision(Action.ANSWER)

    def guard_answer(self, answer: str) -> tuple[str, bool]:
        """Replace an answer that states a dose; returns (answer, was_replaced)."""
        if self.dose_in_answer.search(answer):
            return self.responses[Action.DOSE], True
        return answer, False
