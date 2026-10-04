"""Deterministic merged retrieval corpus and leakage-free SFT preparation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from acharya.config import Settings, canonical_json, sha256_bytes, sha256_file
from acharya.ingest.dedupe import DuplicateComponent, deduplicate
from acharya.ingest.download import Acquisition
from acharya.ingest.loaders import (
    LoadedDataset,
    SourceRecord,
    extract_pdf_pages,
    load_csv_dataset,
    load_text_source,
    parse_qa_pages,
)
from acharya.ingest.normalize import Chunk, make_chunks
from acharya.safety import SafetyPolicy


@dataclass(frozen=True)
class Preparation:
    fingerprint: str
    processed_path: Path
    report_path: Path
    retrieval_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, dict[str, int]]
    hashes: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "processed_path": str(self.processed_path),
            "report_path": str(self.report_path),
            "retrieval_count": self.retrieval_count,
            "split_counts": self.split_counts,
            "source_counts": self.source_counts,
            "hashes": self.hashes,
        }


def _json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json(row) + b"\n")


def _split(chunk: Chunk) -> str:
    bucket = int(sha256_bytes(chunk.provenance.encode())[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _sft_row(chunk: Chunk) -> dict[str, object]:
    return {
        "id": chunk.chunk_id,
        "messages": [
            {"role": "user", "content": chunk.question},
            {"role": "assistant", "content": chunk.text},
        ],
        "provenance": {
            "dataset": chunk.source,
            "version": chunk.dataset_version,
            "locator": chunk.locator or chunk.provenance,
            "license": chunk.license,
            "url": chunk.url,
            "source_reference": chunk.source_reference,
        },
    }


def _duplicate_drops(components: tuple[DuplicateComponent, ...]) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    near: set[str] = set()
    for component in components:
        target = exact if component.kind == "exact" else near
        target.update(set(component.member_ids) - {component.winner_id})
    return exact, near


def prepare(settings: Settings, acquisitions: tuple[Acquisition, ...]) -> Preparation:
    configured = settings.datasets["kaggle"]["datasets"]
    supplemental = settings.datasets.get("supplemental", {})
    text_sources = settings.datasets.get("text_sources", {})
    acquired = {item.dataset_id: item for item in acquisitions}
    if set(acquired) != set(configured):
        raise RuntimeError("preparation requires all configured Kaggle datasets")
    pdf = settings.ingestion["pdf"]
    pdf_records = parse_qa_pages(
        extract_pdf_pages(settings.pdf_path, str(pdf["sha256"]), int(pdf["expected_pages"]))
    )
    records: list[SourceRecord] = list(pdf_records)
    loaded: dict[str, LoadedDataset] = {}
    inventory: list[dict[str, object]] = []
    for dataset_id, raw_spec in configured.items():
        spec = dict(raw_spec)
        acquisition = acquired[dataset_id]
        source = acquisition.extracted / str(spec["primary_member"])
        result = load_csv_dataset(source, dataset_id, spec)
        if not result.records:
            raise RuntimeError(f"zero accepted retrieval rows:{dataset_id}")
        loaded[dataset_id] = result
        records.extend(result.records)
        inventory.append(
            {
                "dataset_id": dataset_id,
                "version": int(str(spec["version"])),
                "archive_sha256": acquisition.archive_sha256,
                "members": acquisition.member_sha256,
            }
        )
    active_supplemental: dict[str, dict[str, object]] = {}
    for dataset_id, raw_spec in supplemental.items():
        spec = dict(raw_spec)
        source = settings.workspace / str(spec["relative_path"])
        expected_member = str(spec["members"][spec["primary_member"]])
        if not source.is_file() or sha256_file(source) != expected_member:
            if not bool(spec.get("required", False)):
                continue
            raise RuntimeError(f"supplemental source hash mismatch:{dataset_id}")
        result = load_csv_dataset(source, dataset_id, spec)
        if not result.records:
            raise RuntimeError(f"zero accepted supplemental rows:{dataset_id}")
        loaded[dataset_id] = result
        active_supplemental[dataset_id] = spec
        records.extend(result.records)
        inventory.append(
            {
                "dataset_id": dataset_id,
                "version": int(str(spec["version"])),
                "archive_sha256": str(spec["archive_sha256"]),
                "members": {str(spec["primary_member"]): expected_member},
            }
        )
    active_text_sources: dict[str, dict[str, object]] = {}
    for dataset_id, raw_spec in text_sources.items():
        spec = dict(raw_spec)
        source = settings.workspace / str(spec["relative_path"])
        expected = str(spec["sha256"])
        if not source.is_file() or sha256_file(source) != expected:
            if not bool(spec.get("required", False)):
                continue
            raise RuntimeError(f"text source hash mismatch:{dataset_id}")
        result = load_text_source(source, dataset_id, spec)
        loaded[dataset_id] = result
        active_text_sources[dataset_id] = spec
        records.extend(result.records)
        inventory.append(
            {
                "dataset_id": dataset_id,
                "version": int(str(spec["version"])),
                "archive_sha256": expected,
                "members": {str(spec["relative_path"]): expected},
            }
        )
    policy = SafetyPolicy(settings.safety)
    chunking = settings.ingestion["chunking"]
    all_chunks = make_chunks(
        tuple(records), policy, int(chunking["max_words"]), int(chunking["overlap_words"])
    )
    role_rejected = tuple(item for item in all_chunks if not policy.role_allowed(item.role))
    eligible = tuple(item for item in all_chunks if policy.role_allowed(item.role))
    dedupe = settings.ingestion["dedupe"]
    unique, components = deduplicate(
        eligible, float(dedupe["jaccard_threshold"]), int(dedupe["shingle_words"])
    )
    if not unique:
        raise RuntimeError("merged corpus has no eligible chunks")
    sources = {item.source for item in unique}
    required_sources = {
        "bundled_pdf",
        *configured.keys(),
        *active_supplemental.keys(),
        *active_text_sources.keys(),
    }
    if sources != required_sources:
        missing = sorted(required_sources - sources)
        raise RuntimeError(f"sources_without_retrieval_rows:{','.join(missing)}")
    sft_sources = set(settings.datasets.get("sft_sources", ()))
    if not sft_sources:
        sft_sources = set(configured) | {"bundled_pdf"}
    sft_chunks = tuple(
        item
        for item in unique
        if item.source in sft_sources and policy.candidate_allowed(item.text)
    )
    split_rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    split_ids: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for chunk in sft_chunks:
        split = _split(chunk)
        split_rows[split].append(_sft_row(chunk))
        split_ids[split].add(chunk.chunk_id)
    if any(not rows for rows in split_rows.values()):
        raise RuntimeError("inadequate SFT split groups")
    if any(
        split_ids[left] & split_ids[right]
        for left in split_ids
        for right in split_ids
        if left != right
    ):
        raise RuntimeError("SFT split leakage")
    exact_drops, near_drops = _duplicate_drops(components)
    rejection_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for item in role_rejected:
        rejection_by_source[item.source][f"role_{item.role}"] += 1
    for dataset_id, result in loaded.items():
        rejection_by_source[dataset_id].update(item.reason for item in result.rejected)
    source_counts: dict[str, dict[str, int]] = {}
    raw_rows = {
        "bundled_pdf": len(pdf_records),
        **{key: value.input_count for key, value in loaded.items()},
    }
    unique_provenance = {item.provenance for item in unique}
    exact_provenance = {item.provenance for item in eligible if item.chunk_id in exact_drops}
    near_provenance = {item.provenance for item in eligible if item.chunk_id in near_drops}
    for source in sorted(required_sources):
        source_records = [item for item in records if item.source == source]
        accepted = sum(item.provenance in unique_provenance for item in source_records)
        remaining = [item for item in source_records if item.provenance not in unique_provenance]
        exact = sum(item.provenance in exact_provenance for item in remaining)
        remaining = [item for item in remaining if item.provenance not in exact_provenance]
        near = sum(item.provenance in near_provenance for item in remaining)
        role_or_policy_rejected = sum(item.provenance not in near_provenance for item in remaining)
        loader_rejected = len(loaded[source].rejected) if source in loaded else 0
        rejected = loader_rejected + role_or_policy_rejected
        input_count = raw_rows[source]
        source_counts[source] = {
            "input": input_count,
            "accepted_unique": accepted,
            "exact_dropped": exact,
            "near_dropped": near,
            "rejected": rejected,
            "retrieval_chunks": sum(item.source == source for item in unique),
        }
        if input_count != accepted + exact + near + rejected:
            raise RuntimeError(f"count reconciliation failed:{source}")
    chunks = [item.as_dict() for item in unique]
    observed = {
        dataset_id: {
            "headers": list(result.headers),
            "header_sha256": result.header_sha256,
            "input_count": result.input_count,
        }
        for dataset_id, result in sorted(loaded.items())
    }
    fingerprint_input = {
        "schema_version": 3,
        "config_sha256": settings.config_hash("ingestion", "safety", "datasets"),
        "raw_inventory": inventory,
        "chunks": chunks,
        "duplicates": [item.as_dict() for item in components],
        "split_ids": {key: sorted(value) for key, value in split_ids.items()},
        "sft_sha256": {
            key: sha256_bytes(canonical_json(value)) for key, value in split_rows.items()
        },
        "source_counts": source_counts,
    }
    fingerprint = sha256_bytes(canonical_json(fingerprint_input))
    processed_root = settings.workspace / "data" / "processed"
    reports_root = settings.workspace / "artifacts" / "preparation"
    processed_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    processed_stage = Path(tempfile.mkdtemp(prefix=".prepared-", dir=processed_root))
    report_stage = Path(tempfile.mkdtemp(prefix=".report-", dir=reports_root))
    try:
        corpus = {"fingerprint": fingerprint, **fingerprint_input}
        _json(processed_stage / "corpus.json", corpus)
        _jsonl(processed_stage / "corpus.jsonl", chunks)
        table = pa.Table.from_pylist(chunks)
        pq.write_table(
            table, processed_stage / "corpus.parquet", compression="zstd", write_statistics=True
        )
        for split, rows in split_rows.items():
            _jsonl(
                processed_stage / f"{split}.jsonl", sorted(rows, key=lambda item: str(item["id"]))
            )
        _json(report_stage / "observed-schemas.json", observed)
        _json(
            report_stage / "counts.json",
            {
                "sources": source_counts,
                "rejection_reasons": {
                    key: dict(sorted(value.items()))
                    for key, value in sorted(rejection_by_source.items())
                },
            },
        )
        _json(report_stage / "duplicate-components.json", [item.as_dict() for item in components])
        _json(
            report_stage / "split-report.json",
            {
                "counts": {key: len(value) for key, value in split_rows.items()},
                "ids": {key: sorted(value) for key, value in split_ids.items()},
                "leakage": False,
            },
        )
        lengths = sorted(len(item.text.split()) for item in sft_chunks)
        _json(
            report_stage / "token-report.json",
            {
                "tokenizer": "deterministic_whitespace_proxy_v1",
                "count": len(lengths),
                "minimum": lengths[0],
                "maximum": lengths[-1],
                "mean": sum(lengths) / len(lengths),
            },
        )
        _json(report_stage / "raw-inventory.json", inventory)
        attribution = [
            {
                "dataset": "bundled_pdf",
                "version": None,
                "license": settings.datasets["bundled_pdf"]["license"],
                "url": None,
                "required_credit": "Repository-provided PDF",
            }
        ]
        attribution.extend(
            {
                "dataset": dataset_id,
                "version": int(str(spec["version"])),
                "license": spec["license"],
                "url": spec["url"],
                "required_credit": spec["required_credit"],
            }
            for dataset_id, spec in sorted(configured.items())
        )
        attribution.extend(
            {
                "dataset": dataset_id,
                "version": int(str(spec["version"])),
                "license": spec["license"],
                "url": spec["url"],
                "required_credit": spec["required_credit"],
            }
            for dataset_id, spec in sorted(active_supplemental.items())
        )
        attribution.extend(
            {
                "dataset": dataset_id,
                "version": int(str(spec["version"])),
                "license": spec["license"],
                "url": spec["url"],
                "required_credit": spec["required_credit"],
            }
            for dataset_id, spec in sorted(active_text_sources.items())
        )
        _json(report_stage / "attribution.json", attribution)
        (report_stage / "DATA_CARD.md").write_text(
            "# AcharyaGPT prepared corpus\n\n"
            "Educational retrieval data; not medical advice or proof of clinical truth. "
            "Dataset-level licenses and attribution are recorded in `attribution.json`. "
            "Treatment, diagnosis, unsafe, unauthenticated, duplicate, and malformed "
            "material is filtered according to committed policy.\n",
            encoding="utf-8",
        )
        hash_targets = sorted(
            [*processed_stage.iterdir(), *report_stage.iterdir()], key=lambda item: item.name
        )
        hashes = {
            item.name: sha256_file(item) for item in hash_targets if item.name != "SHA256SUMS"
        }
        (report_stage / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
            encoding="utf-8",
        )
        hashes["SHA256SUMS"] = sha256_file(report_stage / "SHA256SUMS")
        _json(
            report_stage / "manifest.json",
            {
                "fingerprint": fingerprint,
                "retrieval_count": len(unique),
                "split_counts": {key: len(value) for key, value in split_rows.items()},
                "hashes": hashes,
            },
        )
        processed_target = processed_root / fingerprint
        report_target = reports_root / fingerprint
        if processed_target.exists():
            shutil.rmtree(processed_stage)
        else:
            os.replace(processed_stage, processed_target)
        if report_target.exists():
            shutil.rmtree(report_stage)
        else:
            os.replace(report_stage, report_target)
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        pointer = settings.state_dir / ".active_corpus.json.tmp"
        _json(pointer, {"fingerprint": fingerprint, "path": str(processed_target)})
        os.replace(pointer, settings.state_dir / "active_corpus.json")
        prep_pointer = settings.state_dir / ".active_preparation.json.tmp"
        _json(
            prep_pointer,
            {
                "fingerprint": fingerprint,
                "processed_path": str(processed_target),
                "report_path": str(report_target),
            },
        )
        os.replace(prep_pointer, settings.state_dir / "active_preparation.json")
    except BaseException:
        if processed_stage.exists():
            shutil.rmtree(processed_stage)
        if report_stage.exists():
            shutil.rmtree(report_stage)
        raise
    return Preparation(
        fingerprint,
        processed_root / fingerprint,
        reports_root / fingerprint,
        len(unique),
        {key: len(value) for key, value in split_rows.items()},
        source_counts,
        hashes,
    )


def verify_preparation(settings: Settings, *, strict: bool = True) -> dict[str, object]:
    pointer_path = settings.state_dir / "active_preparation.json"
    if not pointer_path.is_file():
        raise RuntimeError("active preparation is missing")
    pointer: dict[str, Any] = json.loads(pointer_path.read_text(encoding="utf-8"))
    processed = Path(str(pointer["processed_path"]))
    report = Path(str(pointer["report_path"]))
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != pointer["fingerprint"]:
        raise RuntimeError("preparation fingerprint mismatch")
    hashes: dict[str, str] = manifest["hashes"]
    for name, expected in hashes.items():
        path = processed / name if (processed / name).is_file() else report / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"preparation hash mismatch:{name}")
    if strict and any(
        int(manifest["split_counts"][name]) < 1 for name in ("train", "validation", "test")
    ):
        raise RuntimeError("preparation contains an empty split")
    return {
        "fingerprint": pointer["fingerprint"],
        "verified": True,
        "retrieval_count": manifest["retrieval_count"],
        "split_counts": manifest["split_counts"],
    }
