"""Exact text normalization and role-isolated chunking."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from acharya.config import canonical_json
from acharya.safety import SafetyPolicy

if TYPE_CHECKING:
    from acharya.ingest.loaders import SourceRecord

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    record_ordinal: int
    chunk_ordinal: int
    question: str
    text: str
    role: str
    provenance: str
    dataset_version: int | None = None
    locator: str | None = None
    license: str | None = None
    url: str | None = None
    trust_tier: str | None = None
    source_reference: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "record_ordinal": self.record_ordinal,
            "chunk_ordinal": self.chunk_ordinal,
            "question": self.question,
            "text": self.text,
            "role": self.role,
            "provenance": self.provenance,
            "dataset_version": self.dataset_version,
            "locator": self.locator,
            "license": self.license,
            "url": self.url,
            "trust_tier": self.trust_tier,
            "source_reference": self.source_reference,
        }


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u00ad", "")
    value = _SPACE.sub(" ", value).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", value)
    return value


def _chunk_words(text: str, max_words: int, overlap_words: int) -> tuple[str, ...]:
    words = text.split()
    if not words:
        return ()
    step = max_words - overlap_words
    if max_words < 1 or step < 1:
        raise ValueError("chunk sizes must be positive and overlap less than maximum")
    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + max_words]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + max_words >= len(words):
            break
    return tuple(chunks)


def make_chunks(
    records: tuple[SourceRecord, ...],
    policy: SafetyPolicy,
    max_words: int,
    overlap_words: int,
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for record in records:
        for chunk_ordinal, text in enumerate(
            _chunk_words(record.answer, max_words, overlap_words), start=1
        ):
            role = record.role_hint or policy.classify_role(text)
            identity = {
                "provenance": record.provenance,
                "chunk_ordinal": chunk_ordinal,
                "question": record.question,
                "text": text,
                "role": role,
            }
            if record.dataset_version is not None:
                identity["source"] = record.source
                identity["dataset_version"] = record.dataset_version
            chunk_id = "chk_" + hashlib.sha256(canonical_json(identity)).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=record.source,
                    page=record.page,
                    record_ordinal=record.ordinal,
                    chunk_ordinal=chunk_ordinal,
                    question=record.question,
                    text=text,
                    role=role,
                    provenance=record.provenance,
                    dataset_version=record.dataset_version,
                    locator=record.locator,
                    license=record.license,
                    url=record.url,
                    trust_tier=record.trust_tier,
                    source_reference=record.source_reference,
                )
            )
    return tuple(sorted(chunks, key=lambda item: item.chunk_id))
