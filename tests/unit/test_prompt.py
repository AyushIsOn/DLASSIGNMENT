from __future__ import annotations

from pathlib import Path

from acharya.providers.base import ProviderContext
from acharya.rag.prompt import PromptBudget, render_grounded_prompt
from acharya.schemas import ChatMessage


def test_prompt_budget_is_deterministic_and_renumbers_contexts(project_root: Path) -> None:
    contexts = (
        ProviderContext("a", "pdf", 1, "q", "short context", "educational"),
        ProviderContext("b", "pdf", 2, "q", "x" * 500, "educational"),
        ProviderContext("c", "pdf", 3, "q", "other context", "educational"),
    )
    history = (
        ChatMessage(role="user", content="old question"),
        ChatMessage(role="assistant", content="old answer"),
    )
    template = project_root / "prompts" / "grounded_v1.txt"
    base = render_grounded_prompt(
        template,
        "question",
        history,
        contexts,
        PromptBudget(900, 100, 1200),
    )
    repeated = render_grounded_prompt(
        template,
        "question",
        history,
        contexts,
        PromptBudget(900, 100, 1200),
    )
    assert base.utf8 == repeated.utf8
    assert base.prompt_tokens + base.output_tokens <= 1200
    assert "incomplete" not in base.text
    assert base.context_ids == ("a", "c")
    assert "[1] source=pdf" in base.text and "[2] source=pdf" in base.text
    assert "[3]" not in base.text
