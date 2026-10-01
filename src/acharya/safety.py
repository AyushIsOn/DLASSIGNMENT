"""Single compiled medical-safety policy for serving and data preparation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from acharya.config import ConfigurationError, load_yaml


class SafetyAction(StrEnum):
    URGENT = "urgent"
    SELF_HARM = "self_harm"
    DIAGNOSIS = "diagnosis"
    INDIVIDUAL_TREATMENT = "individual_treatment"
    DOSE = "dose"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    response: str | None

    @property
    def retrieves(self) -> bool:
        return self.action is SafetyAction.RETRIEVE


def normalize_safety_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if part.strip())


class SafetyPolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self.allowed_roles = frozenset(str(role) for role in config["allowed_roles"])
        self.precedence = tuple(SafetyAction(item) for item in config["precedence"])
        self.query_patterns = self._compile_group(config["query_patterns"])
        self.candidate_patterns = self._compile_group(config["candidate_patterns"])
        self.role_patterns = self._compile_group(config["role_patterns"])
        responses = config["responses"]
        self.urgent_response = self._fixed_response(responses["urgent"])
        self.refusal_response = self._fixed_response(responses["refusal"])
        self.abstention_response = self._fixed_response(responses["abstention"])

    @classmethod
    def load(cls, path: Path) -> SafetyPolicy:
        return cls(load_yaml(path))

    @staticmethod
    def _compile_group(groups: dict[str, list[str]]) -> dict[str, tuple[re.Pattern[str], ...]]:
        return {
            name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
            for name, patterns in groups.items()
        }

    @staticmethod
    def _fixed_response(record: dict[str, str]) -> str:
        text = record["text"]
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != record["sha256"]:
            raise ConfigurationError("fixed safety response hash mismatch")
        return text

    def classify_query(self, text: str) -> SafetyDecision:
        sentences = tuple(normalize_safety_text(item) for item in _sentences(text))
        matches: set[SafetyAction] = set()
        for name, patterns in self.query_patterns.items():
            if any(pattern.search(sentence) for sentence in sentences for pattern in patterns):
                matches.add(SafetyAction(name))
        for action in self.precedence:
            if action not in matches:
                continue
            response = (
                self.urgent_response
                if action in {SafetyAction.URGENT, SafetyAction.SELF_HARM}
                else self.refusal_response
            )
            return SafetyDecision(action, response)
        return SafetyDecision(SafetyAction.RETRIEVE, None)

    def classify_role(self, text: str) -> str:
        normalized = normalize_safety_text(text)
        for role, patterns in self.role_patterns.items():
            if any(pattern.search(normalized) for pattern in patterns):
                return role
        return "educational"

    def role_allowed(self, role: str) -> bool:
        return role in self.allowed_roles

    def candidate_allowed(self, text: str) -> bool:
        normalized = normalize_safety_text(text)
        return not any(
            pattern.search(normalized)
            for patterns in self.candidate_patterns.values()
            for pattern in patterns
        )
