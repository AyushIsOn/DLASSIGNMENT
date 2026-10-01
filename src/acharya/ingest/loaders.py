"""Strict source loaders with stable, complete provenance."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from acharya.config import ConfigurationError, sha256_bytes, sha256_file
from acharya.ingest.normalize import normalize_text

_Q_MARKER = re.compile(r"(?:^|\n)\s*Q\s*[.:]\s*", re.IGNORECASE)
_A_MARKER = re.compile(r"(?:^|\n|\s)A\s*[.:]\s*", re.IGNORECASE)
_PLACEHOLDERS = frozenset({"", "-", "n/a", "na", "none", "null", "unknown", "not available"})


@dataclass(frozen=True)
class SourceRecord:
    source: str
    page: int
    ordinal: int
    question: str
    answer: str
    dataset_version: int | None = None
    locator: str | None = None
    license: str | None = None
    url: str | None = None
    trust_tier: str | None = None
    role_hint: str | None = None
    source_reference: str | None = None
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def provenance(self) -> str:
        if self.locator is not None:
            return f"{self.source}:version={self.dataset_version}:locator={self.locator}"
        return f"{self.source}:page={self.page}:qa={self.ordinal}"


@dataclass(frozen=True)
class RejectedRow:
    row: int
    reason: str


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: str
    headers: tuple[str, ...]
    header_sha256: str
    input_count: int
    records: tuple[SourceRecord, ...]
    rejected: tuple[RejectedRow, ...]


def header_sha256(headers: tuple[str, ...] | list[str]) -> str:
    payload = json.dumps(list(headers), ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def extract_pdf_pages(path: Path, expected_sha256: str, expected_pages: int = 6) -> tuple[str, ...]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ConfigurationError(f"bundled PDF hash mismatch: {actual_hash}")
    reader = PdfReader(path)
    if len(reader.pages) != expected_pages:
        raise ConfigurationError(f"expected {expected_pages} PDF pages, found {len(reader.pages)}")
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout")
        if not text or not text.strip():
            raise ConfigurationError("bundled PDF contains an empty page")
        pages.append(text.replace("\x00", ""))
    return tuple(pages)


def parse_qa_pages(pages: tuple[str, ...], source: str = "bundled_pdf") -> tuple[SourceRecord, ...]:
    records: list[SourceRecord] = []
    ordinal = 0
    for page_number, page_text in enumerate(pages, start=1):
        parts = _Q_MARKER.split(page_text)
        for part in parts[1:]:
            answer_match = _A_MARKER.search(part)
            if answer_match is None:
                continue
            question = normalize_text(part[: answer_match.start()])
            answer = normalize_text(part[answer_match.end() :])
            if not question or not answer:
                continue
            ordinal += 1
            records.append(SourceRecord(source, page_number, ordinal, question, answer))
    if not records:
        raise ConfigurationError("no Q/A records parsed from bundled PDF")
    return tuple(records)


def _required(value: str, reason: str) -> str:
    normalized = normalize_text(value)
    if normalized.casefold() in _PLACEHOLDERS:
        raise ValueError(reason)
    return normalized


def _decimal(value: str, name: str, floor: Decimal) -> str:
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid_{name}") from error
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise ValueError(f"invalid_{name}")
    if parsed < floor:
        raise ValueError(f"below_{name}_floor")
    return format(parsed, "f")


def _medquad(mapped: dict[str, str], admission: dict[str, Any]) -> tuple[str, str, str | None]:
    question = _required(mapped["question"], "missing_question")
    content = _required(mapped["content"], "missing_content")
    if not int(admission["question_min"]) <= len(question) <= int(admission["question_max"]):
        raise ValueError("question_length")
    if not int(admission["content_min"]) <= len(content) <= int(admission["content_max"]):
        raise ValueError("content_length")
    return question, content, normalize_text(mapped["source_reference"])


def _knowledge(mapped: dict[str, str], admission: dict[str, Any]) -> tuple[str, str, str]:
    title = _required(mapped["title"], "missing_title")
    if not int(admission["title_min"]) <= len(title) <= int(admission["title_max"]):
        raise ValueError("title_length")
    source = _required(mapped["source_reference"], "missing_source_reference")
    candidates = [
        mapped["modern_equivalent"],
        mapped["body_system"],
        mapped["dosha"],
        mapped["prognosis"],
        mapped["symptoms"],
    ]
    substantive = [
        normalize_text(value)
        for value in candidates
        if len(normalize_text(value)) >= int(admission["substantive_min"])
    ]
    if not substantive:
        raise ValueError("missing_substantive_content")
    content = ". ".join([f"Ayurvedic name: {title}", *substantive])
    return title, content, source


def _healthcare(mapped: dict[str, str], admission: dict[str, Any]) -> tuple[str, str, str]:
    problem = _required(mapped["problem"], "missing_problem")
    source = _required(mapped["source_reference"], "missing_source_reference")
    mapped["confidence"] = _decimal(
        mapped["confidence"], "confidence", Decimal(str(admission["confidence_floor"]))
    )
    mapped["auth_score"] = _decimal(
        mapped["auth_score"], "auth_score", Decimal(str(admission["auth_score_floor"]))
    )
    notes = _required(mapped["authentication_notes"], "missing_authentication_notes")
    folded = notes.casefold()
    placeholders = {str(item).casefold() for item in admission["authentication_placeholders"]}
    if folded in placeholders:
        raise ValueError("placeholder_authentication_notes")
    if any(
        str(phrase).casefold() in folded for phrase in admission["negative_authentication_phrases"]
    ):
        raise ValueError("negative_authentication_notes")
    content_fields = (
        "symptoms",
        "preventive_advice",
        "seasonal_suitability",
        "classical_texts",
        "modern_evidence",
        "contraindications",
    )
    content = ". ".join(
        normalize_text(mapped[name]) for name in content_fields if normalize_text(mapped[name])
    )
    if not content:
        raise ValueError("missing_substantive_content")
    return problem, content, source


def load_csv_dataset(
    path: Path, dataset_id: str, spec: dict[str, Any], *, enforce_row_count: bool = True
) -> LoadedDataset:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        expected = tuple(str(item) for item in spec["headers"])
        observed_hash = header_sha256(headers)
        if headers != expected:
            raise ConfigurationError(f"schema_header_mismatch:{dataset_id}")
        if observed_hash != str(spec["header_sha256"]):
            raise ConfigurationError(f"schema_hash_mismatch:{dataset_id}")
        rows = list(reader)
    if enforce_row_count and len(rows) != int(spec["expected_rows"]):
        raise ConfigurationError(f"row_count_mismatch:{dataset_id}:{len(rows)}")
    records: list[SourceRecord] = []
    rejected: list[RejectedRow] = []
    field_map = {str(key): str(value) for key, value in spec["field_map"].items()}
    for row_number, row in enumerate(rows, start=2):
        mapped = {
            target: normalize_text(row.get(header, "") or "")
            for header, target in field_map.items()
        }
        try:
            if dataset_id == "gpreda/medquad/1":
                question, answer, source_reference = _medquad(mapped, spec["admission"])
                role_hint = None
            elif dataset_id == "akashkumarpr/ayurvedic-knowledge-dataset/1":
                question, answer, source_reference = _knowledge(mapped, spec["admission"])
                role_hint = "educational"
            elif dataset_id == "aliainaanraza/ayurveda-healthcare-dataset/2":
                question, answer, source_reference = _healthcare(mapped, spec["admission"])
                role_hint = "educational"
            else:
                raise ConfigurationError(f"dataset_not_allowlisted:{dataset_id}")
        except ValueError as error:
            rejected.append(RejectedRow(row_number, str(error)))
            continue
        records.append(
            SourceRecord(
                dataset_id,
                0,
                row_number - 1,
                question,
                answer,
                int(spec["version"]),
                f"row={row_number}",
                str(spec["license"]),
                str(spec["url"]),
                str(spec["trust_tier"]),
                role_hint,
                source_reference,
                mapped,
            )
        )
    return LoadedDataset(
        dataset_id, headers, observed_hash, len(rows), tuple(records), tuple(rejected)
    )
