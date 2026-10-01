"""Deterministic, network-independent extractive answer provider."""

from __future__ import annotations

import re

from acharya.providers.base import ProviderContext, ProviderResult
from acharya.rag.index import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveProvider:
    name = "extractive"

    def __init__(self, max_sentences: int = 2, max_characters: int = 700) -> None:
        self.max_sentences = max_sentences
        self.max_characters = max_characters

    def answer(self, query: str, contexts: tuple[ProviderContext, ...]) -> ProviderResult | None:
        query_terms = set(tokenize(query))
        ranked: list[tuple[int, int, str, ProviderContext]] = []
        for context_index, context in enumerate(contexts):
            for sentence_index, sentence in enumerate(_SENTENCE.split(context.text)):
                clean = sentence.strip()
                if not clean or len(clean) > self.max_characters:
                    continue
                overlap = len(query_terms & set(tokenize(clean)))
                if overlap:
                    ranked.append((-overlap, context_index * 1000 + sentence_index, clean, context))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[3].chunk_id))
        selected = ranked[: self.max_sentences]
        answer_parts: list[str] = []
        used: list[str] = []
        total = 0
        for _, _, sentence, context in selected:
            added = len(sentence) + (1 if answer_parts else 0)
            if total + added > self.max_characters:
                continue
            answer_parts.append(sentence)
            total += added
            if context.chunk_id not in used:
                used.append(context.chunk_id)
        if not answer_parts:
            return None
        return ProviderResult(" ".join(answer_parts), tuple(used), self.name)
