"""Configuration, paths, device helpers and small JSON utilities."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """configs/train.yaml, or the file named by ACHARYA_CONFIG (used by the tests)."""
        path = path or os.environ.get("ACHARYA_CONFIG") or ROOT / "configs/train.yaml"
        return cls(yaml.safe_load(Path(path).read_text()))

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    @property
    def model_dir(self) -> Path:
        return self.path(os.environ.get("ACHARYA_BASE_MODEL_DIR",
                                        self.raw["base_model"]["local_dir"]))

    @property
    def data_dir(self) -> Path:
        return self.path(self.raw["data_dir"])

    @property
    def output_dir(self) -> Path:
        return self.path(os.environ.get("ACHARYA_RUN_DIR", self.raw["output_dir"]))

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapter"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @property
    def lora(self) -> dict[str, Any]:
        return dict(self.raw["lora"])

    @property
    def train(self) -> dict[str, Any]:
        return dict(self.raw["train"])

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.raw["evaluation"])


def write_json(path: Path, value: Any) -> None:
    """Atomic JSON write (a crash never leaves a half-written file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"time": time.strftime("%H:%M:%S"), "event": event, **fields},
                     default=str), flush=True)


def device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gib=round(props.total_memory / 2**30, 1),
                    bf16=torch.cuda.is_bf16_supported())
    return info


def cpu_allowed() -> bool:
    """ACHARYA_ALLOW_CPU=1 lets the CPU test-suite run the GPU scripts with a tiny model."""
    return os.environ.get("ACHARYA_ALLOW_CPU") == "1"


def use_bf16() -> bool:
    """bf16 on any bf16-capable GPU (A100/H100/H200). ACHARYA_CPU_BF16=1 lets CPU tests
    exercise the same bf16 code path."""
    import torch

    if torch.cuda.is_available():
        return bool(torch.cuda.is_bf16_supported())
    return os.environ.get("ACHARYA_CPU_BF16") == "1"


def model_dtype() -> Any:
    import torch

    return torch.bfloat16 if use_bf16() else torch.float32


def load_base_model(model_dir: Path, *, for_training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": model_dtype(), "local_files_only": True,
                              "attn_implementation": "sdpa"}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.config.use_cache = not for_training
    return model


def load_tokenizer(model_dir: Path, *, padding_side: str = "right") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = padding_side
    return tokenizer


def stop_token_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")]
    return [int(i) for i in ids if i is not None and i >= 0]
