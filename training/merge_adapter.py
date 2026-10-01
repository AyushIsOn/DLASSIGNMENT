"""Optional external bf16 adapter merge; never an implicit training fallback."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from acharya.config import canonical_json, sha256_file
from acharya.lightning import LOCKED_REVISION, PreflightError, QLoRAProfile


def merge(workspace: Path, adapter: Path, destination: Path) -> dict[str, object]:
    QLoRAProfile.load(workspace)
    manifest = adapter / "ADAPTER_MANIFEST.json"
    if not manifest.is_file():
        raise PreflightError("hash-locked adapter manifest is required")
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise PreflightError("optional bf16 merge requires CUDA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".merged-", dir=destination.parent))
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, torch_dtype=torch.bfloat16, device_map={"": "cpu"}, trust_remote_code=False
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(stage, safe_serialization=True, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(adapter, trust_remote_code=False).save_pretrained(stage)
        files = {
            path.name: sha256_file(path)
            for path in sorted(stage.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        result: dict[str, object] = {
            "schema_version": 1,
            "base_revision": LOCKED_REVISION,
            "adapter_manifest_sha256": sha256_file(manifest),
            "files": files,
            "dtype": "bfloat16",
            "optional": True,
        }
        (stage / "MERGED_MANIFEST.json").write_bytes(canonical_json(result) + b"\n")
        if destination.exists():
            raise PreflightError("merge destination already exists")
        os.replace(stage, destination)
        return result
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(args.workspace.resolve(), args.adapter.resolve(), args.destination.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
