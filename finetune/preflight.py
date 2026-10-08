"""GPU preflight, run at the start of the GPU session (takes seconds).

Fails fast - before any expensive step - if the GPU, CUDA, bf16, memory, disk,
model files or dataset are not as expected.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from finetune.common import Config, cpu_allowed, device_info, log
from finetune.data import verify_manifest
from finetune.download_model import verified


def check(config: Config, require_gpu: bool = True) -> dict[str, object]:
    problems: list[str] = []
    info: dict[str, object] = {}
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=30)
        info["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    elif require_gpu:
        problems.append("nvidia-smi not found: this machine has no NVIDIA GPU. "
                        "Switch the Studio to an H200 (or A100/H100 80 GB).")
    import torch

    info.update(device_info())
    if require_gpu:
        if not torch.cuda.is_available():
            problems.append("PyTorch cannot see the GPU (torch.cuda.is_available() is False). "
                            f"nvidia-smi says: {info.get('nvidia_smi')}")
        else:
            gib = torch.cuda.get_device_properties(0).total_memory / 2**30
            if gib < 70:
                problems.append(f"GPU has {gib:.0f} GiB; this config needs >= 80 GB "
                                "(H200/H100/A100-80GB)")
            if not torch.cuda.is_bf16_supported():
                problems.append("GPU does not support bf16 (need Ampere or newer)")
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            info["bf16_matmul_ok"] = bool(torch.isfinite((a @ a).float()).all())
    free_gib = shutil.disk_usage(config.output_dir.parent if config.output_dir.parent.exists()
                                 else Path.cwd()).free / 2**30
    info["free_disk_gib"] = round(free_gib, 1)
    model_gib = sum((config.model_dir / name).stat().st_size for name in
                    config.raw["base_model"]["shards"] if (config.model_dir / name).is_file()
                    ) / 2**30
    # checkpoints (3 kept, ~3 GB each for 14B r=64) + recovery tar + evaluation
    need_training = 12 + 0.6 * model_gib
    info["needed_disk_gib"] = {"training": round(need_training), "merge": round(model_gib + 2)}
    if free_gib < need_training:
        problems.append(f"only {free_gib:.0f} GiB free disk; training needs >= "
                        f"{need_training:.0f} GiB (checkpoints + recovery archive). Delete "
                        "round-1 leftovers, e.g. rm -rf artifacts/run/merged artifacts/gguf")
    elif free_gib < need_training + model_gib + 2:
        info["warning"] = (f"enough disk to train and evaluate, but the merged model needs "
                           f"~{model_gib + 2:.0f} GiB more")
    if not verified(config.model_dir, config):
        problems.append(f"base model in {config.model_dir} is missing or unverified: run "
                        "`bash scripts/lightning_cpu_setup.sh` (on CPU) first")
    try:
        verify_manifest(config.data_dir)
    except Exception as error:
        problems.append(str(error))
    info["problems"] = problems
    log("preflight", **info)
    if problems:
        raise SystemExit("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="skip the GPU checks")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config.load()
    require_gpu = not (args.cpu or cpu_allowed())
    print(json.dumps(check(config, require_gpu=require_gpu), indent=2, default=str))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
