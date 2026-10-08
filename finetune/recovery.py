"""Portable recovery archives, so a run can continue on another machine or Lightning account.

During training, after a checkpoint is saved (at most every `pack_every_minutes`), a
background `tar` writes <run dir>/recovery/recovery-latest.tar with:
    checkpoints/checkpoint-<latest>/   adapter + optimizer + scheduler + RNG + trainer state
    checkpoints/checkpoint-<best>/     (if different) so "best validation loss" survives
    RUN_INFO.json, metrics.jsonl       run identity + the loss curve so far
It is written to a temporary name and renamed, so the file on disk is always complete.

If ACHARYA_HUB_REPO (e.g. "your-name/acharya-recovery", a PRIVATE model repo) and HF_TOKEN
are set, each archive is also uploaded there; that is the easiest way to move a run between
Lightning accounts without downloading 6 GB through the browser.

Restore with:  bash scripts/restore_and_resume.sh <recovery-latest.tar | hub>
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from finetune.common import log

ARCHIVE = "recovery-latest.tar"


class Packer:
    def __init__(self, run_dir: Path, every_minutes: float | None = None) -> None:
        self.run_dir = run_dir
        self.dir = run_dir / "recovery"
        minutes = every_minutes if every_minutes is not None else float(
            os.environ.get("ACHARYA_PACK_EVERY_MINUTES", "8"))
        self.every = minutes * 60
        self.last = 0.0
        self.process: subprocess.Popen[bytes] | None = None
        self.disabled = os.environ.get("ACHARYA_NO_RECOVERY") == "1"

    def pack(self, step: int, best: str | None, force: bool = False) -> None:
        if self.disabled or (not force and time.time() - self.last < self.every):
            return
        self.wait()  # never two tars at once
        checkpoints = self.run_dir / "checkpoints"
        members = [f"checkpoints/checkpoint-{step}"]
        if best and Path(best).name != f"checkpoint-{step}" \
                and (checkpoints / Path(best).name).is_dir():
            members.append(f"checkpoints/{Path(best).name}")
        if not (checkpoints / f"checkpoint-{step}").is_dir():
            return
        members += [name for name in ("RUN_INFO.json", "metrics.jsonl")
                    if (self.run_dir / name).is_file()]
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = self.dir / f".{ARCHIVE}.tmp"
        final = self.dir / ARCHIVE
        script = (f'tar -cf "{temporary}" -C "{self.run_dir}" ' + " ".join(
            f'"{member}"' for member in members) + f' && mv -f "{temporary}" "{final}"')
        repo = os.environ.get("ACHARYA_HUB_REPO")
        if repo and os.environ.get("HF_TOKEN"):
            script += (f' && "{_python()}" -m finetune.recovery upload "{final}" "{repo}"'
                       f' >> "{self.dir / "upload.log"}" 2>&1')
        self.process = subprocess.Popen(["bash", "-c", script], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, start_new_session=True)
        self.last = time.time()
        log("recovery_archive", step=step, members=members, path=final,
            upload=bool(repo and os.environ.get("HF_TOKEN")))

    def wait(self) -> None:
        if self.process is not None:
            code = self.process.wait()
            if code != 0:
                log("recovery_archive_failed", exit_code=code,
                    note="training continues; the previous archive is kept")
            self.process = None


def _python() -> str:
    import sys

    return sys.executable


def fix_checkpoint_paths(checkpoints: Path) -> None:
    """Checkpoints restored on another machine record the old absolute path of the best
    checkpoint; point it at this machine (Trainer uses it to protect the best checkpoint
    from rotation and to load it at the end)."""
    for state_path in checkpoints.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text())
        best = state.get("best_model_checkpoint")
        if not best:
            continue
        local = checkpoints / Path(best).name
        fixed = str(local) if local.is_dir() else None
        if fixed != best:
            state["best_model_checkpoint"] = fixed
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            log("checkpoint_path_fixed", file=state_path, best=fixed)


def upload(archive: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE, repo_id=repo,
                    commit_message=f"recovery archive {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("recovery_uploaded", repo=repo)


def download(repo: str, target: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, ARCHIVE, local_dir=str(target))
    return Path(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload")
    up.add_argument("archive", type=Path)
    up.add_argument("repo")
    down = sub.add_parser("download")
    down.add_argument("repo")
    down.add_argument("target", type=Path)
    args = parser.parse_args()
    if args.command == "upload":
        upload(args.archive, args.repo)
    else:
        print(download(args.repo, args.target))


if __name__ == "__main__":
    main()
