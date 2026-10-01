"""External PEFT-backed RAG smoke with support-gated serving evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.lightning import LOCKED_REVISION, PreflightError
from acharya.providers.peft_local import PEFTLocalProvider
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


def smoke(workspace: Path, adapter: Path, output: Path) -> dict[str, object]:
    manifest = adapter / "ADAPTER_MANIFEST.json"
    if not manifest.is_file():
        raise PreflightError("adapter manifest is required for PEFT RAG smoke")
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise PreflightError("PEFT RAG smoke requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=False)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=False
    )
    invocations = 0
    generated_answer_hashes: list[str] = []

    def generate(prompt: str, output_tokens: int) -> str:
        nonlocal invocations
        invocations += 1
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        generated = model.generate(**encoded, max_new_tokens=output_tokens, do_sample=False)
        answer = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        generated_answer_hashes.append(sha256_bytes(answer.encode()))
        return answer

    provider = PEFTLocalProvider(
        adapter_path=adapter,
        manifest_path=manifest,
        base_revision=LOCKED_REVISION,
        generator=generate,
    )
    service = RAGService(workspace, generative_provider=provider, generation_mode="optional")
    response = service.chat(ChatRequest(message="Which doshas are involved in Eka kusta?"))
    response_answer_sha256 = sha256_bytes(response.answer.encode())
    record: dict[str, object] = {
        "schema_version": 2,
        "provider": "peft-local",
        "adapter_manifest_sha256": sha256_file(manifest),
        "provider_invoked": invocations > 0,
        "generated_candidate_count": len(generated_answer_hashes),
        "generated_candidate_accepted": response_answer_sha256 in generated_answer_hashes,
        "response_sha256": sha256_bytes(canonical_json(response.model_dump(mode="json"))),
        "response_answer_sha256": response_answer_sha256,
        "outcome": response.outcome,
        "citation_count": len(response.citations),
        "support_gate_active": service.generative_ready,
        "quality_gain_claimed": False,
    }
    if (
        not record["provider_invoked"]
        or not record["support_gate_active"]
        or not record["generated_candidate_accepted"]
        or response.outcome != "answered"
        or not response.citations
    ):
        raise PreflightError("PEFT provider was not accepted behind the RAG support gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(record) + b"\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            smoke(args.workspace.resolve(), args.adapter.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
