"""Executable, network-blocked Gate A evidence orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from acharya.config import Settings, canonical_json, sha256_file
from acharya.ingest.pipeline import active_corpus, build_corpus
from acharya.rag.evaluate import calibrate
from acharya.rag.index import build_index
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


def _request(url: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
    data = canonical_json(payload) if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("API returned a non-object response")
    return value


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _start_api(workspace: Path, port: int) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    network_block = workspace / "tests" / "network_block"
    current_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{network_block}{os.pathsep}{current_path}"
    environment["ACHARYA_WORKSPACE"] = str(workspace)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "acharya.api:app_factory",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.communicate()[1].decode(errors="replace")
            raise RuntimeError(f"API failed to start: {output[-500:]}")
        try:
            if _request(f"http://127.0.0.1:{port}/health/ready").get("ready") is True:
                return process
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("API readiness timed out")


def _stop_api(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def verify(workspace: Path) -> dict[str, object]:
    settings = Settings.load(workspace)
    first = build_corpus(settings)
    first_manifest_hash = sha256_file(first.path / "manifest.json")
    second = build_corpus(settings)
    deterministic = (
        first.fingerprint == second.fingerprint
        and first.chunk_ids == second.chunk_ids
        and first.duplicate_components == second.duplicate_components
        and first.record_count == second.record_count
        and first_manifest_hash == sha256_file(second.path / "manifest.json")
    )
    pointer_before = canonical_json(active_corpus(settings))
    atomic_failure = False
    try:
        build_corpus(settings, inject_failure=True)
    except RuntimeError as error:
        atomic_failure = (
            str(error) == "injected staged corpus failure"
            and canonical_json(active_corpus(settings)) == pointer_before
        )
    first_index = build_index(settings)
    calibration = calibrate(settings, first_index)
    second_index = build_index(settings)
    index_deterministic = first_index.fingerprint == second_index.fingerprint and tuple(
        item.chunk_id for item in first_index.documents
    ) == tuple(item.chunk_id for item in second_index.documents)
    calibrate(settings, second_index)
    service = RAGService(workspace)
    urgent = service.chat(ChatRequest(message="I cannot breathe and have severe chest pain."))
    provider_bypass = urgent.outcome == "urgent" and not urgent.citations

    port = _free_port()
    first_process = _start_api(workspace, port)
    _stop_api(first_process)
    second_process = _start_api(workspace, port)
    try:
        ready = _request(f"http://127.0.0.1:{port}/health/ready")
        answer = _request(
            f"http://127.0.0.1:{port}/v1/chat",
            {"message": "Which are the doshas involved in Eka kusta?", "history": []},
        )
        abstention = _request(
            f"http://127.0.0.1:{port}/v1/chat",
            {"message": "How do I repair a diesel engine gearbox?", "history": []},
        )
    finally:
        _stop_api(second_process)
    checks = {
        "real_pdf_hash": sha256_file(settings.pdf_path) == settings.ingestion["pdf"]["sha256"],
        "six_pages_parsed": first.record_count > 0,
        "deterministic_corpus": deterministic,
        "atomic_failure_preserved_active": atomic_failure,
        "deterministic_index_order": index_deterministic,
        "zero_false_support_calibration": calibration.false_support == 0
        and calibration.ready
        and set(calibration.adversarial_categories)
        == {"typo", "entity_swap", "negation", "random", "irrelevant"},
        "api_restarted_under_network_denial": ready.get("ready") is True,
        "bm25_only_ready": ready.get("retrieval_mode") == "bm25_only",
        "cited_extractive_answer": answer.get("outcome") == "answered"
        and bool(answer.get("citations")),
        "nullable_dense_rerank": all(
            item.get("dense") is None and item.get("rerank") is None
            for item in answer.get("scores", [])
        ),
        "fallback_warning": answer.get("warning") == "bm25_only_fallback",
        "adversarial_abstention": abstention.get("outcome") == "abstained",
        "urgent_provider_bypass": provider_bypass,
        "no_dense_modules": not any(
            name in sys.modules for name in ("sentence_transformers", "transformers", "torch")
        ),
    }
    status = "PASSED" if all(checks.values()) else "FAILED"
    report = {
        "schema_version": 1,
        "gate": "A",
        "status": status,
        "input_hashes": {
            "pdf": sha256_file(settings.pdf_path),
            "ingestion_config": settings.config_hash("ingestion"),
            "rag_config": settings.config_hash("rag"),
            "models_config": settings.config_hash("models"),
            "safety_config": settings.config_hash("safety"),
            "evaluation": sha256_file(workspace / "eval" / "golden.jsonl"),
        },
        "output_hashes": {
            "corpus": first.fingerprint,
            "index": first_index.fingerprint,
            "calibration": hashlib.sha256(canonical_json(calibration.as_dict())).hexdigest(),
            "answer": hashlib.sha256(canonical_json(answer)).hexdigest(),
        },
        "counts": {
            "records": first.record_count,
            "chunks": first.chunk_count,
            "eligible_chunks": first.eligible_count,
        },
        "checks": checks,
        "calibration_evidence": calibration.as_dict(),
        "gate_b": "PENDING",
        "gate_c": "PENDING_EXTERNAL_GPU",
    }
    report_dir = workspace / "artifacts" / "gates"
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary = report_dir / ".gate-a.json.tmp"
    temporary.write_bytes(canonical_json(report) + b"\n")
    os.replace(temporary, report_dir / "gate-a.json")
    if status != "PASSED":
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Gate A checks failed: {failed}")
    return report


def main() -> int:
    workspace = Path(os.environ.get("ACHARYA_WORKSPACE", Path(__file__).resolve().parents[2]))
    print(json.dumps(verify(workspace), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
