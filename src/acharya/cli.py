"""Command-line entry points for corpus, preparation, retrieval, and serving."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from acharya.config import Settings, canonical_json
from acharya.ingest.pipeline import build_corpus
from acharya.preparation import prepare_handoff, verify_handoff
from acharya.providers.base import ProviderError, ProviderRequest
from acharya.providers.kiro_cli import KiroCLIProvider
from acharya.rag.evaluate import calibrate, calibrate_support
from acharya.rag.index import build_index
from acharya.rag.prompt import RenderedPrompt
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


def _workspace(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _workspace_argument(command: argparse.ArgumentParser) -> None:
    command.add_argument("--workspace", dest="command_workspace", type=_workspace)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="acharya")
    result.add_argument("--workspace", type=_workspace)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("build-corpus", "build-index", "calibrate", "calibrate-support", "status"):
        _workspace_argument(commands.add_parser(name))
    prepare_command = commands.add_parser("prepare-handoff")
    _workspace_argument(prepare_command)
    prepare_command.add_argument("--strict", action="store_true")
    verify = commands.add_parser("verify-preparation")
    _workspace_argument(verify)
    verify.add_argument("--strict", action="store_true")
    index = commands.add_parser("index")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    index_build = index_commands.add_parser("build")
    _workspace_argument(index_build)
    index_build.add_argument(
        "--retrieval-mode", choices=("bm25_only", "mvp_hybrid", "full"), required=True
    )
    index_build.add_argument("--allow-bm25-fallback", action="store_true")
    evaluate = commands.add_parser("evaluate")
    _workspace_argument(evaluate)
    evaluate.add_argument(
        "--retrieval-mode", choices=("bm25_only", "mvp_hybrid", "full"), required=True
    )
    evaluate.add_argument("--golden", type=_workspace)
    evaluate.add_argument("--activate-calibration", action="store_true")
    provider_check = commands.add_parser("provider-check")
    _workspace_argument(provider_check)
    provider_check.add_argument("--provider", choices=("kiro-cli",), required=True)
    provider_check.add_argument("--env-file", type=_workspace, required=True)
    provider_check.add_argument("--record", type=_workspace, required=True)
    smoke = commands.add_parser("smoke")
    _workspace_argument(smoke)
    smoke.add_argument("query")
    serve = commands.add_parser("serve")
    _workspace_argument(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--provider", choices=("extractive", "ollama", "peft-local"), default="extractive"
    )
    serve.add_argument("--model")
    serve.add_argument("--adapter", type=_workspace)
    return result


def _load_provider_environment(path: Path) -> dict[str, str]:
    allowed = {"KIRO_API_KEY", "KIRO_BASE_URL", "KIRO_MODEL", "KIRO_CONTEXT_WINDOW_TOKENS"}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        clean = value.strip()
        if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
            clean = clean[1:-1]
        values[key] = clean
    return values


def _provider_check(env_file: Path, record_path: Path) -> tuple[dict[str, object], int]:
    prompt_id = "kiro_api_ok_v1"
    status = "UNVERIFIED"
    category = "external_provider_unverified"
    exit_code: int | None = None
    cli_version = "2.26.0"
    try:
        values = _load_provider_environment(env_file)
        provider = KiroCLIProvider(environment=values)
        cli_version = provider.version
        text = "KIRO_API_OK"
        prompt = RenderedPrompt(text, text.encode(), len(text.encode()), 16, ())
        result = provider.generate(ProviderRequest(prompt, time.monotonic() + 30.0))
        exit_code = 0
        if result.text == text:
            status = "PASSED"
            category = "passed_exact_response"
        else:
            category = "response_mismatch"
    except FileNotFoundError:
        status = "BLOCKED_EXTERNAL_PROVIDER"
        category = "env_file_missing"
    except ProviderError as error:
        status = "BLOCKED_EXTERNAL_PROVIDER"
        category = error.category
        exit_code = error.exit_code
    except (OSError, ValueError, RuntimeError):
        status = "BLOCKED_EXTERNAL_PROVIDER"
        category = "external_provider_failure"
    record: dict[str, object] = {
        "schema_version": 1,
        "provider": "kiro-cli",
        "cli_version": cli_version,
        "prompt_id": prompt_id,
        "status": status,
        "category": category,
        "exit_code": exit_code,
        "checked_at": datetime.now(UTC).isoformat(),
        "credential_recorded": False,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_name(f".{record_path.name}.tmp")
    temporary.write_bytes(canonical_json(record) + b"\n")
    os.replace(temporary, record_path)
    return record, 0 if status == "PASSED" else 2


def _update_provider_gate(settings: Settings, record: dict[str, object]) -> None:
    path = settings.workspace / "artifacts" / "gates" / "gate-b.json"
    value: dict[str, object] = {"schema_version": 1, "gate": "B", "status": "PENDING"}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                value = loaded
        except (OSError, ValueError, TypeError):
            pass
    components = value.get("components")
    if not isinstance(components, dict):
        components = {}
    components["provider"] = {
        key: record[key]
        for key in (
            "provider",
            "cli_version",
            "prompt_id",
            "status",
            "category",
            "exit_code",
            "checked_at",
            "credential_recorded",
        )
    }
    value["components"] = components
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workspace = args.command_workspace or args.workspace or Path.cwd()
    settings = Settings.load(workspace)
    return_code = 0
    if args.command == "build-corpus":
        output: object = build_corpus(settings).__dict__
    elif args.command == "build-index":
        output = build_index(settings).as_dict()
    elif args.command == "calibrate":
        output = calibrate(settings).as_dict()
    elif args.command == "calibrate-support":
        output = calibrate_support(settings).as_dict()
    elif args.command == "prepare-handoff":
        output = prepare_handoff(settings, strict=bool(args.strict)).as_dict()
    elif args.command == "verify-preparation":
        output = verify_handoff(settings, strict=bool(args.strict))
    elif args.command == "index":
        output = build_index(
            settings, args.retrieval_mode, allow_bm25_fallback=bool(args.allow_bm25_fallback)
        ).as_dict()
    elif args.command == "evaluate":
        index_value = __import__("acharya.rag.index", fromlist=["load_index"]).load_index(
            settings, args.retrieval_mode
        )
        output = calibrate(
            settings, index_value, golden=args.golden, activate=bool(args.activate_calibration)
        ).as_dict()
    elif args.command == "status":
        output = [item.model_dump(mode="json") for item in RAGService(workspace).gate_statuses()]
    elif args.command == "provider-check":
        output, return_code = _provider_check(args.env_file, args.record)
        _update_provider_gate(settings, output)
    elif args.command == "smoke":
        output = RAGService(workspace).chat(ChatRequest(message=args.query)).model_dump(mode="json")
    elif args.command == "serve":
        import os

        os.environ["ACHARYA_WORKSPACE"] = str(workspace)
        os.environ["ACHARYA_PROVIDER"] = args.provider
        if args.model:
            os.environ["ACHARYA_MODEL"] = args.model
        if args.adapter:
            os.environ["ACHARYA_ADAPTER_PATH"] = str(args.adapter)
        uvicorn.run("acharya.api:app_factory", factory=True, host=args.host, port=args.port)
        return 0
    else:
        raise RuntimeError("unhandled command")
    print(json.dumps(output, default=str, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
