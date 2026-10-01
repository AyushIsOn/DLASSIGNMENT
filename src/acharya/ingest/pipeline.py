"""Staged deterministic corpus construction with atomic activation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acharya.config import Settings, canonical_json, sha256_bytes, sha256_file
from acharya.ingest.dedupe import deduplicate
from acharya.ingest.loaders import extract_pdf_pages, parse_qa_pages
from acharya.ingest.normalize import make_chunks
from acharya.safety import SafetyPolicy


@dataclass(frozen=True)
class CorpusBuild:
    fingerprint: str
    path: Path
    record_count: int
    chunk_count: int
    eligible_count: int
    duplicate_components: tuple[dict[str, object], ...]
    chunk_ids: tuple[str, ...]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def active_corpus(settings: Settings) -> dict[str, Any] | None:
    pointer = settings.state_dir / "active_corpus.json"
    return _read_json(pointer) if pointer.is_file() else None


def build_corpus(settings: Settings, *, inject_failure: bool = False) -> CorpusBuild:
    pdf = settings.ingestion["pdf"]
    pages = extract_pdf_pages(settings.pdf_path, str(pdf["sha256"]), int(pdf["expected_pages"]))
    records = parse_qa_pages(pages)
    policy = SafetyPolicy(settings.safety)
    chunking = settings.ingestion["chunking"]
    all_chunks = make_chunks(
        records,
        policy,
        int(chunking["max_words"]),
        int(chunking["overlap_words"]),
    )
    eligible = tuple(item for item in all_chunks if policy.role_allowed(item.role))
    dedupe = settings.ingestion["dedupe"]
    unique, components = deduplicate(
        eligible, float(dedupe["jaccard_threshold"]), int(dedupe["shingle_words"])
    )
    chunks = [item.as_dict() for item in unique]
    component_values = [item.as_dict() for item in components]
    fingerprint_input = {
        "schema_version": 1,
        "pdf_sha256": sha256_file(settings.pdf_path),
        "config_sha256": settings.config_hash("ingestion", "safety"),
        "records": [
            {
                "source": item.source,
                "page": item.page,
                "ordinal": item.ordinal,
                "question": item.question,
                "answer": item.answer,
            }
            for item in records
        ],
        "chunks": chunks,
        "duplicate_components": component_values,
    }
    fingerprint = sha256_bytes(canonical_json(fingerprint_input))
    corpus = {"fingerprint": fingerprint, **fingerprint_input}
    processed_root = settings.workspace / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".corpus-", dir=processed_root))
    try:
        _write_json(staging / "corpus.json", corpus)
        with (staging / "corpus.jsonl").open("wb") as handle:
            for chunk in chunks:
                handle.write(canonical_json(chunk) + b"\n")
        _write_json(
            staging / "manifest.json",
            {
                "fingerprint": fingerprint,
                "corpus_sha256": sha256_file(staging / "corpus.json"),
                "jsonl_sha256": sha256_file(staging / "corpus.jsonl"),
                "record_count": len(records),
                "chunk_count": len(all_chunks),
                "eligible_count": len(unique),
            },
        )
        if inject_failure:
            raise RuntimeError("injected staged corpus failure")
        target = processed_root / fingerprint
        if target.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, target)
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        pointer_temp = settings.state_dir / ".active_corpus.json.tmp"
        _write_json(pointer_temp, {"fingerprint": fingerprint, "path": str(target)})
        os.replace(pointer_temp, settings.state_dir / "active_corpus.json")
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return CorpusBuild(
        fingerprint=fingerprint,
        path=processed_root / fingerprint,
        record_count=len(records),
        chunk_count=len(all_chunks),
        eligible_count=len(unique),
        duplicate_components=tuple(component_values),
        chunk_ids=tuple(item.chunk_id for item in unique),
    )
