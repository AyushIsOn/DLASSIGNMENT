"""Deterministic grounded prompt construction and provider budgeting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from acharya.providers.base import ProviderContext
from acharya.schemas import ChatMessage


@dataclass(frozen=True)
class PromptBudget:
    context_window_tokens: int
    output_tokens: int
    maximum_prompt_bytes: int = 131_072

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("token budgets must be positive")
        if self.output_tokens >= self.context_window_tokens:
            raise ValueError("output reservation must be smaller than the context window")
        if self.maximum_prompt_bytes <= 0:
            raise ValueError("prompt byte budget must be positive")


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    utf8: bytes
    prompt_tokens: int
    output_tokens: int
    contexts: tuple[ProviderContext, ...]

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.contexts)


def conservative_token_count(text: str) -> int:
    """Count UTF-8 bytes conservatively when a provider tokenizer is unavailable."""
    return len(text.encode("utf-8"))


def _complete_history(
    history: Sequence[ChatMessage],
) -> tuple[tuple[ChatMessage, ChatMessage], ...]:
    if len(history) % 2:
        raise ValueError("history must contain complete user/assistant pairs")
    pairs: list[tuple[ChatMessage, ChatMessage]] = []
    for index in range(0, len(history), 2):
        user, assistant = history[index], history[index + 1]
        if user.role != "user" or assistant.role != "assistant":
            raise ValueError("history must alternate user and assistant roles")
        pairs.append((user, assistant))
    return tuple(pairs)


def _render(
    instructions: str,
    query: str,
    history: Sequence[tuple[ChatMessage, ChatMessage]],
    contexts: Sequence[ProviderContext],
) -> str:
    history_text = "\n".join(
        f"USER: {user.content}\nASSISTANT: {assistant.content}" for user, assistant in history
    )
    context_text = "\n\n".join(
        f"[{number}] source={context.source} page={context.page}\n{context.text}"
        for number, context in enumerate(contexts, 1)
    )
    return (
        f"{instructions.rstrip()}\n\n"
        f"HISTORY:\n{history_text or '(none)'}\n\n"
        f"CONTEXTS:\n{context_text or '(none)'}\n\n"
        f"QUESTION:\n{query.strip()}\n\nANSWER:\n"
    )


def render_grounded_prompt(
    template_path: Path,
    query: str,
    history: Sequence[ChatMessage],
    contexts: Sequence[ProviderContext],
    budget: PromptBudget,
    *,
    token_count: Callable[[str], int] = conservative_token_count,
) -> RenderedPrompt:
    instructions = template_path.read_text(encoding="utf-8")
    pairs = _complete_history(history)
    selected_history: list[tuple[ChatMessage, ChatMessage]] = []
    selected_contexts: list[ProviderContext] = []

    def fits(
        candidate_history: Sequence[tuple[ChatMessage, ChatMessage]],
        candidate_contexts: Sequence[ProviderContext],
    ) -> bool:
        candidate = _render(instructions, query, candidate_history, candidate_contexts)
        return (
            len(candidate.encode("utf-8")) <= budget.maximum_prompt_bytes
            and token_count(candidate) + budget.output_tokens <= budget.context_window_tokens
        )

    if not fits((), ()):
        raise ValueError("prompt instructions and question exceed provider budget")
    for pair in reversed(pairs):
        candidate = [pair, *selected_history]
        if fits(candidate, ()):
            selected_history = candidate
    for context in contexts:
        candidate_contexts = [*selected_contexts, context]
        if fits(selected_history, candidate_contexts):
            selected_contexts = candidate_contexts
    text = _render(instructions, query, selected_history, selected_contexts)
    encoded = text.encode("utf-8")
    return RenderedPrompt(
        text=text,
        utf8=encoded,
        prompt_tokens=token_count(text),
        output_tokens=budget.output_tokens,
        contexts=tuple(selected_contexts),
    )


def correction_prompt(prompt: RenderedPrompt) -> RenderedPrompt:
    suffix = (
        "\nCORRECTION: The previous candidate was rejected. Return a fully grounded replacement; "
        "every claim must cite only its supporting numbered context.\nANSWER:\n"
    )
    text = prompt.text + suffix
    return RenderedPrompt(
        text=text,
        utf8=text.encode("utf-8"),
        prompt_tokens=conservative_token_count(text),
        output_tokens=prompt.output_tokens,
        contexts=prompt.contexts,
    )
