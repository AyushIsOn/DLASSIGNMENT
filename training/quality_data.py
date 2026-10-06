"""Copy/OCR audits and qualified explanatory QA data; never rewrite active data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.lightning import PreflightError
from acharya.rag.grounding import GroundingError, parse_generated_claims
from acharya.safety import SafetyPolicy


def context_text(row: dict[str, Any]) -> str:
    user = str(next(message["content"] for message in row["messages"] if message["role"] == "user"))
    if "CONTEXTS:\n" not in user or "\n\nQUESTION:" not in user:
        raise PreflightError("row lacks a distinct grounding context")
    contexts = user.split("CONTEXTS:\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    return contexts.split("\n", 1)[1].strip()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def copy_fraction(answer: str, context: str, width: int = 5) -> float:
    target, source = words(answer), words(context)
    if len(target) < width:
        return float(target == source)
    source_grams = {tuple(source[i : i + width]) for i in range(len(source) - width + 1)}
    grams = [tuple(target[i : i + width]) for i in range(len(target) - width + 1)]
    return sum(gram in source_grams for gram in grams) / len(grams)


def audit_row(row: dict[str, Any]) -> list[str]:
    context = context_text(row)
    answer = row["messages"][-1]["content"]
    reasons = []
    try:
        claims = parse_generated_claims(answer)
        if any(claim.citation_numbers != (1,) for claim in claims):
            reasons.append("target_cites_unavailable_context")
    except GroundingError:
        reasons.append("target_has_missing_or_malformed_claim_citations")
    if copy_fraction(answer, context) >= 0.65:
        reasons.append("target_is_mostly_source_copy")
    if row.get("task_type") == "historical_extraction":
        reasons.append("historical_extraction_requires_source_review")
    if re.search(r"[<>^{}]|\b\w*\d\w+\b", context):
        reasons.append("possible_OCR_or_markup_artifact")
    if len(words(answer)) < 12 or len(words(answer)) > 180:
        reasons.append("answer_length_outside_12_to_180_words")
    return reasons


def audit(directory: Path, output: Path) -> dict[str, Any]:
    records, counts = [], {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (directory / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        counts[split] = len(rows)
        for row in rows:
            try:
                reasons = audit_row(row)
            except (KeyError, ValueError, StopIteration) as error:
                reasons = [f"invalid_row:{type(error).__name__}"]
            records.append({"id": row.get("id"), "split": split, "reasons": reasons})
    report = {
        "counts": counts,
        "rejected_count": sum(bool(r["reasons"]) for r in records),
        "records": records,
        "training_ready": False,
        "note": "Passing these filters alone does not establish answer quality.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(report) + b"\n")
    return report


def compile_reviewed(candidates: Path, output: Path, benchmark: Path) -> dict[str, Any]:
    """Compile separately reviewed targets, enforcing provenance and held-out isolation."""
    if output.exists():
        raise PreflightError("quality output already exists; use a fresh directory")
    held_out = {
        tuple(words(json.loads(line)["query"]))
        for line in benchmark.read_text().splitlines()
        if line.strip()
    }
    policy = SafetyPolicy.load(Path(__file__).resolve().parents[1] / "configs/safety.yaml")
    splits: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "validation", "test")}
    ids: set[str] = set()
    questions: dict[tuple[str, ...], str] = {}
    source_hashes: dict[str, str] = {}
    review_evidence = []
    for line in candidates.read_text().splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("review_status") != "approved" or not candidate.get("reviewer"):
            raise PreflightError("every candidate needs a named reviewer and approved status")
        if not all(
            candidate.get("review_checks", {}).get(key) is True
            for key in ("factuality", "relevance", "clarity", "safety")
        ):
            raise PreflightError("all four review criteria must be approved")
        row, split = candidate["row"], candidate["split"]
        answer = row["messages"][-1]["content"]
        if not policy.candidate_allowed(answer):
            raise PreflightError("unsafe training target")
        claims = parse_generated_claims(answer)
        if any(claim.citation_numbers != (1,) for claim in claims):
            raise PreflightError("target must cite its single source context")
        if split not in splits or not row.get("provenance"):
            raise PreflightError("missing split or source provenance")
        if audit_row(row):
            raise PreflightError(f"candidate failed copy/OCR/length audit:{row['id']}")
        question = tuple(words(row["question"]))
        if question in held_out or row["id"] in ids or question in questions:
            raise PreflightError("duplicate or independent-benchmark question in candidates")
        digest = sha256_bytes(" ".join(words(context_text(row))).encode())
        if digest in source_hashes and source_hashes[digest] != split:
            raise PreflightError("source-context leakage across splits")
        ids.add(row["id"])
        questions[question] = split
        source_hashes[digest] = split
        splits[split].append(row)
        review_evidence.append(candidate)
    minimum = {"train": 500, "validation": 100, "test": 100}
    if any(len(splits[key]) < minimum[key] for key in splits):
        raise PreflightError(
            "need at least 500 train / 100 validation / 100 test reviewed examples"
        )
    output.mkdir(parents=True)
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_bytes(
            b"".join(canonical_json(row) + b"\n" for row in rows)
        )
    (output / "REVIEW_EVIDENCE.json").write_bytes(canonical_json(review_evidence) + b"\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "training_ready": True,
        "clinical_review_completed": False,
        "counts": {key: len(value) for key, value in splits.items()},
        "benchmark_sha256": sha256_file(benchmark),
        "candidate_sha256": sha256_file(candidates),
        "files": {f"{split}.jsonl": sha256_file(output / f"{split}.jsonl") for split in splits},
    }
    manifest["files"]["REVIEW_EVIDENCE.json"] = sha256_file(output / "REVIEW_EVIDENCE.json")
    (output / "QUALITY_MANIFEST.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("compile")
    build.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    report = (
        audit(args.data, args.output)
        if args.command == "audit"
        else compile_reviewed(args.candidates, args.output, args.benchmark)
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
