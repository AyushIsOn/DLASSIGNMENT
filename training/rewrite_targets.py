"""Generate reviewable explanatory targets from clean, non-historical sources.

Uses the locked *base*, never the copying adapter. Drafts cannot be trained until
reviewed and compiled. Incremental JSONL preserves completed drafts on interruption.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import zip_longest
from pathlib import Path

from acharya.config import canonical_json, sha256_bytes
from acharya.lightning import PreflightError
from training.quality_data import audit_row, context_text
from training.runtime import make_generator


def rewrite(workspace: Path, output: Path, limit: int, minutes: float) -> None:
    if limit < 1 or not 1 <= minutes <= 120:
        raise PreflightError("positive limit and a 1-120 minute ceiling are required")
    pointer = json.loads((workspace / "artifacts/state/active_preparation.json").read_text())
    data = Path(pointer["processed_path"])
    if not data.is_absolute():
        data = workspace / data
    rows = [
        (split, json.loads(line))
        for split in ("train", "validation", "test")
        for line in (data / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Interleave splits so a time-limited run cannot consume only train rows.
    grouped = [[item for item in rows if item[0] == split]
               for split in ("validation", "test", "train")]
    rows = [item for batch in zip_longest(*grouped) for item in batch if item is not None]
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            value = json.loads(line)
            if value["preparation_fingerprint"] != pointer["fingerprint"]:
                raise PreflightError("draft file belongs to different input data")
            completed.add(value["source_id"])
    generator = make_generator(workspace, None)
    deadline, generated = time.monotonic() + minutes * 60, 0
    for position, (split, row) in enumerate(rows):
        if generated >= limit or time.monotonic() >= deadline:
            break
        if row["id"] in completed or row.get("task_type") == "historical_extraction":
            continue
        context = context_text(row)
        if "possible_OCR_or_markup_artifact" in audit_row(row):
            continue
        task = (
            "Explain the central concept in plain language and its limits.",
            "Explain a relationship or distinction explicitly supported by the source.",
            "Explain what the source does and does not establish; avoid inventing limitations.",
        )[(position // 3) % 3]
        prompt = (
            "Create an educational question and a concise explanatory answer from the context. "
            "Return only JSON with keys question and answer. Paraphrase; do not copy a record or "
            "list of fields. Do not diagnose or recommend treatments. Cite each claim with [1]. "
            "Explain traditional concepts as theory, not established biomedical fact. "
            "The answer must contain 12-180 words.\n\nHISTORY:\n(none)\n\n"
            f"CONTEXTS:\n[1] source={row['provenance']['dataset']} page=1\n{context}\n\n"
            f"QUESTION:\n{task} Write one original educational question and its supported answer."
            "\n\nANSWER:\n"
        )
        print(json.dumps({"event": "rewrite_started", "source_id": row["id"]}), flush=True)
        raw = generator(prompt, 384)
        try:
            value = json.JSONDecoder().raw_decode(raw.lstrip())[0]
            question, answer = str(value["question"]).strip(), str(value["answer"]).strip()
            if len(question) < 12 or not answer:
                raise ValueError("empty or too short question")
            rewritten = dict(row)
            rewritten.update(
                id="qa_" + sha256_bytes((row["id"] + question).encode())[:24],
                question=question,
                task_type="explanatory_qa",
            )
            system = row["messages"][0]
            user = (
                "HISTORY:\n(none)\n\nCONTEXTS:\n"
                f"[1] source={row['provenance']['dataset']} page=1\n{context}"
                f"\n\nQUESTION:\n{question}\n\nANSWER:\n"
            )
            rewritten["messages"] = [
                system,
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
            reasons = audit_row(rewritten)
            candidate = {
                "source_id": row["id"],
                "split": split,
                "row": rewritten,
                "preparation_fingerprint": pointer["fingerprint"],
                "teacher": "locked-Qwen3-8B-base",
                "draft_task": task,
                "review_status": "pending",
                "reviewer": None,
                "automated_rejection_reasons": reasons,
                "review_checks": {
                    "factuality": False,
                    "relevance": False,
                    "clarity": False,
                    "safety": False,
                },
            }
            with output.open("ab") as handle:
                handle.write(canonical_json(candidate) + b"\n")
            generated += 1
            print(
                json.dumps(
                    {
                        "event": "draft_saved",
                        "count": generated,
                        "automated_rejection_reasons": reasons,
                    }
                ),
                flush=True,
            )
        except (ValueError, KeyError, TypeError) as error:
            print(
                json.dumps({"event": "draft_rejected", "reason": type(error).__name__}), flush=True
            )
    print(
        json.dumps(
            {
                "drafts_written": generated,
                "training_ready": False,
                "next_step": "Review each draft against its context, then compile.",
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--minutes", type=float, default=60)
    args = parser.parse_args()
    rewrite(args.workspace.resolve(), args.output.resolve(), args.limit, args.minutes)


if __name__ == "__main__":
    main()
