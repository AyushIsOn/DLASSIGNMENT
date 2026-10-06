"""Lazy CUDA generation from the verified local base and exported adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from acharya.lightning import LOCKED_REVISION, PreflightError, QLoRAProfile
from acharya.rag.prompt import grounded_chat_messages
from training.train_qlora import _verify_model_snapshot


def make_generator(workspace: Path, adapter: Path | None) -> Callable[[str, int], str]:
    model: Any = None
    tokenizer: Any = None

    def generate(prompt: str, output_tokens: int) -> str:
        nonlocal model, tokenizer
        messages = grounded_chat_messages(prompt)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise PreflightError("PEFT serving requires CUDA; use extractive mode locally")
        if model is None:
            profile = QLoRAProfile.load(workspace)
            snapshot = workspace / "artifacts" / "model-cache" / LOCKED_REVISION
            _verify_model_snapshot(profile, snapshot)
            tokenizer = AutoTokenizer.from_pretrained(adapter or snapshot, local_files_only=True)
            base = AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
            model = (PeftModel.from_pretrained(base, adapter, local_files_only=True)
                     if adapter is not None else base)
            model.eval()
        chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=output_tokens, do_sample=False)
        return str(
            tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        )

    return generate
