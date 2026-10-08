"""`acharya serve` - run the chat API for the iOS app.

GPU (Lightning, right after training):
    acharya serve --backend transformers --host 0.0.0.0 --port 8000
Mac (after `ollama create acharyagpt -f artifacts/gguf/Modelfile`):
    acharya serve --backend ollama --model acharyagpt
Quick check without the server:
    acharya ask "What is the modern equivalent of Amlapitta?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def config_paths() -> tuple[Path, Path]:
    """(base model dir, adapter dir) from configs/train.yaml (or ACHARYA_CONFIG)."""
    from finetune.common import Config

    config = Config.load()
    return config.model_dir, config.adapter_dir


def build_generator(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from acharya.generation import EchoGenerator, OllamaGenerator, TransformersGenerator

    if args.backend == "ollama":
        return OllamaGenerator(args.model or "acharyagpt", args.ollama_url)
    if args.backend == "echo":
        return EchoGenerator()
    defaults = config_paths()
    model_dir = Path(args.model_dir or defaults[0]).expanduser().resolve()
    adapter = None if args.no_adapter else Path(args.adapter or defaults[1]).expanduser().resolve()
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"No adapter at {adapter}. Train first, or pass --no-adapter to "
                         "serve the base model.")
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"No model at {model_dir}")
    return TransformersGenerator(model_dir, adapter)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="acharya", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("serve", "ask"):
        command = commands.add_parser(name)
        command.add_argument("--backend", choices=("transformers", "ollama", "echo"),
                             default="transformers")
        command.add_argument("--model-dir", help="default: base model of configs/train.yaml")
        command.add_argument("--adapter", help="default: <output_dir>/adapter of the config")
        command.add_argument("--no-adapter", action="store_true", help="serve the base model")
        command.add_argument("--model", help="Ollama model name (default: acharyagpt)")
        command.add_argument("--ollama-url", default="http://127.0.0.1:11434")
        command.add_argument("--no-retrieval", action="store_true")
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
        else:
            command.add_argument("question")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from acharya.schemas import ChatRequest
    from acharya.service import ChatService

    generator = build_generator(args)
    service = ChatService(generator, use_retrieval=not args.no_retrieval)
    if args.command == "ask":
        response = service.chat(ChatRequest(message=args.question))
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    import uvicorn

    from acharya.api import create_app

    if hasattr(generator, "load"):
        print(f"Loading {generator.name} ... (first load takes ~1 minute)", file=sys.stderr)
        generator.load()
    ok, detail = generator.ready()
    if not ok:
        print(f"WARNING: model backend not ready: {detail}", file=sys.stderr)
    print(f"AcharyaGPT API on http://{args.host}:{args.port}  (model: {generator.name})",
          file=sys.stderr)
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
