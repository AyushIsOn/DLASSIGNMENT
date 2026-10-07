"""One prompt format shared by dataset building, training, evaluation and serving.

Keeping this in a single module guarantees the model is served with exactly the
format it was trained on. `render_prompt` reproduces Qwen3's official chat
template with thinking disabled (verified against the real tokenizer in tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

SYSTEM_PROMPT = (
    "You are AcharyaGPT, an educational assistant for Ayurveda. Answer accurately and "
    "concisely. If knowledge base entries are given, use the relevant entry and ignore the "
    "others. Never diagnose a person or give medicine doses; for emergencies, advise urgent "
    "medical care."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Qwen3 non-thinking generation prefix (what enable_thinking=False inserts).
ASSISTANT_PREFIX = f"{IM_START}assistant\n<think>\n\n</think>\n\n"
STOP_STRINGS = (IM_END, "<|endoftext|>")


class Message(TypedDict):
    role: str
    content: str


def format_contexts(contexts: Sequence[str]) -> str:
    return "\n".join(f"[{number}] {text}" for number, text in enumerate(contexts, 1))


def user_content(question: str, contexts: Sequence[str] = ()) -> str:
    """Closed-book questions are sent verbatim; open-book ones carry numbered entries."""
    question = question.strip()
    if not contexts:
        return question
    return f"Knowledge base entries:\n{format_contexts(contexts)}\n\nQuestion: {question}"


def build_messages(
    question: str,
    contexts: Sequence[str] = (),
    history: Sequence[Message] = (),
    system: str = SYSTEM_PROMPT,
) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": system}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in history)
    messages.append({"role": "user", "content": user_content(question, contexts)})
    return messages


def render_prompt(messages: Sequence[Message]) -> str:
    """Render ChatML exactly like Qwen3's template with add_generation_prompt=True and
    enable_thinking=False. Earlier assistant turns are rendered without think blocks."""
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("the last message must be a user message")
    parts = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if role == "system" and index != 0:
            raise ValueError("the system message must come first")
        parts.append(f"{IM_START}{role}\n{message['content']}{IM_END}\n")
    parts.append(ASSISTANT_PREFIX)
    return "".join(parts)


def completion_text(answer: str) -> str:
    """Training target appended after `render_prompt`; ends with the Qwen EOS token."""
    return f"{answer.strip()}{IM_END}"


def clean_generation(text: str) -> str:
    """Strip stop tokens and any stray reasoning block from a raw generation."""
    for stop in STOP_STRINGS:
        text = text.split(stop, 1)[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()
