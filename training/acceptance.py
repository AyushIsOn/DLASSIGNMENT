"""Lightning API/RAG acceptance gate before connecting the iOS client.

Reference terms are regression checks, not a substitute for clinical review.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from acharya.config import canonical_json, sha256_file


def check(workspace: Path, adapter: Path, output: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from acharya.api import create_app
    from acharya.lightning import LOCKED_REVISION
    from acharya.providers.peft_local import PEFTLocalProvider
    from acharya.rag.evaluate import load_calibration, load_support_calibration
    from acharya.rag.index import load_index
    from acharya.rag.service import RAGService
    from training.runtime import make_generator

    generator = make_generator(workspace, adapter)
    generated: set[str] = set()

    def observed(prompt: str, tokens: int) -> str:
        value = generator(prompt, tokens)
        generated.add(value.strip())
        return value

    provider = PEFTLocalProvider(
        adapter_path=adapter,
        manifest_path=adapter / "ADAPTER_MANIFEST.json",
        base_revision=LOCKED_REVISION,
        generator=observed,
    )
    service = RAGService(workspace, generative_provider=provider, generation_mode="optional")
    index = load_index(service.settings)
    if index.mode != "full":
        raise RuntimeError("acceptance requires dense + BM25 + reranker retrieval")
    calibration = load_calibration(service.settings, index)
    support = load_support_calibration(service.settings)
    cases_path = workspace / "eval/acceptance.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    records = []
    with TestClient(create_app(workspace, service=service)) as client:
        ready = client.get("/health/ready")
        for case in cases:
            started = time.monotonic()
            response = client.post("/v1/chat", json={"message": case["query"], "history": []})
            elapsed = time.monotonic() - started
            body = response.json()
            answer = str(body.get("answer", ""))
            terms_pass = all(
                term.casefold() in answer.casefold() for term in case["required_terms"]
            )
            passed = (
                response.status_code == 200
                and body.get("outcome") == case["outcome"]
                and terms_pass
                and (case["outcome"] != "answered" or bool(body.get("citations")))
            )
            records.append(
                {
                    **case,
                    "passed": passed,
                    "http_status": response.status_code,
                    "latency_seconds": elapsed,
                    "response": body,
                    "generated_answer_accepted": answer.strip() in generated,
                }
            )
            print(
                json.dumps(
                    {"case": case["id"], "passed": passed, "latency_seconds": round(elapsed, 2)}
                ),
                flush=True,
            )
        invalid = client.post(
            "/v1/chat",
            json={"message": "hello", "history": [{"role": "user", "content": "incomplete"}]},
        )
    latencies = sorted(float(row["latency_seconds"]) for row in records)
    report = {
        "schema_version": 1,
        "engineering_acceptance_passed": (
            ready.status_code == 200
            and invalid.status_code == 422
            and all(row["passed"] for row in records)
            and any(row["generated_answer_accepted"] for row in records)
        ),
        "ready_status": ready.status_code,
        "invalid_history_status": invalid.status_code,
        "model_revision": LOCKED_REVISION,
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "corpus_fingerprint": index.corpus_fingerprint,
        "test_sha256": sha256_file(cases_path),
        "retrieval_mode": index.mode,
        "retrieval_calibration": calibration.as_dict(),
        "support_calibration": support.as_dict(),
        "case_count": len(records),
        "passed_count": sum(row["passed"] for row in records),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "records": records,
        "clinical_accuracy_certified": False,
        "human_review_required": [
            "Compare base/adapter answers in evaluation.json",
            "Check source accuracy and OCR errors",
            "Inspect unsupported claims and omissions",
            "Approve latency on the intended deployment GPU before iOS use",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(report) + b"\n")
    output.with_suffix(".md").write_text(
        "# Lightning acceptance report\n\n"
        f"Engineering gate: {'PASS' if report['engineering_acceptance_passed'] else 'FAIL'}\n\n"
        f"{report['passed_count']}/{len(records)} cases passed. "
        f"Median latency: {statistics.median(latencies):.2f}s.\n\n"
        "Read the JSON for full answers, citations and failure cases. "
        "A passing engineering gate is not clinical approval.\n"
    )
    if not report["engineering_acceptance_passed"]:
        raise RuntimeError(f"acceptance failed; review {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.workspace.resolve(), args.adapter.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
