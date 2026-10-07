"""Download the pinned Qwen3-8B snapshot and verify every weight shard (CPU only).

    python -m finetune.download_model         # ~16.4 GB, resumable

A `.verified.json` marker is written after all SHA-256 hashes match, so later
steps (including the GPU run) do not need to re-hash 16 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from finetune.common import Config, log
from finetune.data import sha256_file

MARKER = ".verified.json"


def verified(model_dir: Path, config: Config) -> bool:
    marker = model_dir / MARKER
    if not marker.is_file():
        return False
    recorded = json.loads(marker.read_text())
    shards = config.raw["base_model"]["shards"]
    if recorded.get("revision") != config.raw["base_model"]["revision"]:
        return False
    for name, digest in shards.items():
        path = model_dir / name
        if recorded.get("shards", {}).get(name) != digest or not path.is_file():
            return False
        if path.stat().st_size != recorded.get("sizes", {}).get(name):
            return False
    return True


def download(config: Config) -> Path:
    from huggingface_hub import snapshot_download

    base = config.raw["base_model"]
    model_dir = config.model_dir
    if verified(model_dir, config):
        log("model_ready", path=model_dir, note="already downloaded and verified")
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            log("downloading", repository=base["repository"], revision=base["revision"],
                attempt=attempt)
            snapshot_download(repo_id=base["repository"], revision=base["revision"],
                              local_dir=model_dir,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
            break
        except Exception as error:  # network hiccups: retry with backoff
            log("download_retry", error=f"{type(error).__name__}: {error}")
            if attempt == 5:
                raise
            time.sleep(10 * attempt)
    sizes, bad = {}, []
    for name, expected in base["shards"].items():
        path = model_dir / name
        log("verifying", shard=name)
        if not path.is_file() or sha256_file(path) != expected:
            bad.append(name)
        else:
            sizes[name] = path.stat().st_size
    if bad:
        for name in bad:
            (model_dir / name).unlink(missing_ok=True)
        raise SystemExit(f"hash mismatch for {bad}; deleted them - run this command again")
    (model_dir / MARKER).write_text(json.dumps(
        {"revision": base["revision"], "shards": base["shards"], "sizes": sizes}, indent=2))
    log("model_ready", path=model_dir)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    download(Config.load(args.config) if args.config else Config.load())


if __name__ == "__main__":
    main()
