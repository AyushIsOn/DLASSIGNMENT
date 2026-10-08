"""Merge the LoRA adapter into the base model and save a standalone bf16 model.

    python -m finetune.export            # -> <output_dir>/merged (~30 GB for 14B)

The merged folder is what llama.cpp converts to GGUF for Ollama on a Mac
(scripts/export_gguf.sh). Merging runs fine on the GPU (seconds) or CPU (minutes).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from finetune.common import Config, load_tokenizer, log


def merge(config: Config, adapter: Path, output: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from finetune.train import adapter_complete

    if not adapter_complete(adapter):
        raise SystemExit(f"No trained adapter in {adapter}")
    if (output / "config.json").is_file() and (output / "MERGED_FROM.json").is_file():
        recorded = json.loads((output / "MERGED_FROM.json").read_text())
        if recorded.get("adapter_mtime") == (adapter / "adapter_model.safetensors").stat().st_mtime:
            log("merge_skipped", reason="already merged", output=output)
            return output
    # bf16 merge: on CPU this needs ~17 GB RAM, on the GPU it takes seconds.
    device_map = {"": 0} if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(
        config.model_dir, dtype=torch.bfloat16, local_files_only=True, device_map=device_map)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
    load_tokenizer(adapter).save_pretrained(staging)
    (staging / "MERGED_FROM.json").write_text(json.dumps({
        "base_model": config.raw["base_model"]["repository"],
        "revision": config.raw["base_model"]["revision"],
        "adapter": str(adapter),
        "adapter_mtime": (adapter / "adapter_model.safetensors").stat().st_mtime,
    }, indent=2))
    if output.exists():
        shutil.rmtree(output)
    os.replace(staging, output)
    log("merged", output=output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config.load()
    merge(config, args.adapter or config.adapter_dir, args.output or config.output_dir / "merged")


if __name__ == "__main__":
    main()
