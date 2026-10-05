"""Independent end-to-end QA comparison; references never enter model prompts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from acharya.config import canonical_json, sha256_file
from acharya.providers.base import ProviderRequest, ProviderResult
from acharya.providers.peft_local import PEFTLocalProvider
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


def concept_match(answer: str, groups: list[list[str]]) -> bool:
    return all(any(term.casefold() in answer.casefold() for term in group) for group in groups)


def evaluate(
    workspace: Path, benchmark: Path, adapter: Path | None, output: Path, mode: str
) -> dict[str, Any]:
    from acharya.lightning import LOCKED_REVISION
    from training.runtime import make_generator

    cases = [json.loads(line) for line in benchmark.read_text().splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate benchmark IDs")
    services = {"extractive": RAGService(workspace)}
    if mode == "gpu":
        if adapter is None:
            raise ValueError("GPU comparison requires an exported adapter")
        generator = make_generator(workspace, None)

        class BaseProvider:
            name = "locked-base-evaluation"

            def generate(self, request: ProviderRequest) -> ProviderResult:
                return ProviderResult(
                    generator(request.prompt.text, request.prompt.output_tokens), (), self.name
                )

        services["base"] = RAGService(
            workspace, generative_provider=BaseProvider(), generation_mode="optional"
        )
        provider = PEFTLocalProvider(
            adapter_path=adapter.resolve(),
            manifest_path=adapter / "ADAPTER_MANIFEST.json",
            base_revision=LOCKED_REVISION,
            generator=make_generator(workspace, adapter),
        )
        services["adapter"] = RAGService(
            workspace, generative_provider=provider, generation_mode="optional"
        )
    output.mkdir(parents=True, exist_ok=False)
    records, reviews, mapping = [], [], []
    rng = random.Random(3407)
    for case in cases:
        responses = []
        for name, service in services.items():
            # The only benchmark content supplied to RAG is the user's question.
            response = service.chat(ChatRequest(message=case["query"], history=[]))
            passed = (
                response.outcome == case["expected_outcome"]
                and concept_match(response.answer, case["concept_groups"])
                and (case["expected_outcome"] != "answered" or bool(response.citations))
            )
            record = {
                "id": case["id"],
                "model": name,
                "screening_passed": passed,
                "response": response.model_dump(mode="json"),
                "failure_stage": getattr(service, "last_failure_reason", None),
                "retrieval_diagnostics": [
                    {
                        "source": hit.document.source,
                        "chunk_id": hit.document.chunk_id,
                        "supported": hit.supported,
                        "bm25": hit.bm25_score,
                        "lexical_coverage": hit.lexical_coverage,
                        "informative_overlap": hit.informative_overlap,
                        "dense": hit.dense_score,
                        "rerank": hit.rerank_score,
                        "role": hit.document.role,
                        "text": hit.document.text[:600],
                    }
                    for hit in getattr(service, "last_retrieval_hits", ())
                ],
            }
            records.append(record)
            responses.append((name, record["response"]))
            print(
                json.dumps({"case": case["id"], "model": name, "screening_passed": passed}),
                flush=True,
            )
        rng.shuffle(responses)
        for number, (name, response) in enumerate(responses):
            blind_id = f"{case['id']}-{number}"
            reviews.append(
                {
                    "blind_id": blind_id,
                    "query": case["query"],
                    "review_notes": case["review_notes"],
                    "response": response,
                    "scores_0_to_2": {
                        "factuality": None,
                        "relevance": None,
                        "clarity": None,
                        "safety": None,
                    },
                    "reviewer": None,
                    "comments": None,
                }
            )
            mapping.append({"blind_id": blind_id, "model": name})
        # Save partial results after every question; preserve progress on interruption.
        (output / "responses.json").write_bytes(canonical_json(records) + b"\n")
        (output / "blind-review.json").write_bytes(canonical_json(reviews) + b"\n")
        (output / "review-key.json").write_bytes(canonical_json(mapping) + b"\n")
    report = {
        "benchmark_sha256": sha256_file(benchmark),
        "case_count": len(cases),
        "models": {
            name: {
                "screening_passed": sum(
                    row["screening_passed"] for row in records if row["model"] == name
                ),
                "case_count": len(cases),
            }
            for name in services
        },
        "answer_quality_established": False,
        "clinical_accuracy_certified": False,
        "human_review_required": True,
        "limitations": [
            "Small project-authored benchmark; not expert-validated.",
            "Concept matching is screening, not correctness grading.",
            "Base and adapter both use production verification/fallback.",
            "Benchmark is post-hoc for the existing adapter.",
        ],
    }
    (output / "summary.json").write_bytes(canonical_json(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.workspace.resolve(),
                args.benchmark.resolve(),
                args.adapter.resolve() if args.adapter else None,
                args.output.resolve(),
                args.mode,
            )
        )
    )


if __name__ == "__main__":
    main()
