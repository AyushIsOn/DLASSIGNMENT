"""External-only Qwen2.5 QLoRA training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from acharya.config import canonical_json, sha256_bytes, sha256_file
from acharya.lightning import (
    LOCKED_REVISION,
    PreflightError,
    QLoRAProfile,
    select_resume_checkpoint,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
                raise PreflightError("training row does not contain messages")
            values.append(value)
    if not values:
        raise PreflightError("training split is empty")
    return values


def _verify_model_snapshot(profile: QLoRAProfile, snapshot: Path) -> None:
    for name, record in profile.value["base_model"]["files"].items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise PreflightError(f"locked model file missing:{name}")
        if sha256_file(path) != record["sha256"]:
            raise PreflightError(f"locked model file hash mismatch:{name}")


def acquire_model(workspace: Path) -> dict[str, object]:
    """Download then hash-verify the locked snapshot without loading model code."""
    profile = QLoRAProfile.load(workspace)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            profile.value["base_model"]["repository"],
            revision=LOCKED_REVISION,
            local_dir=workspace / "artifacts" / "model-cache" / LOCKED_REVISION,
        )
    )
    _verify_model_snapshot(profile, snapshot)
    return {
        "snapshot": str(snapshot),
        "revision": LOCKED_REVISION,
        "files": {
            name: sha256_file(snapshot / name)
            for name in sorted(profile.value["base_model"]["files"])
        },
    }


def train(workspace: Path, output: Path, max_steps: int) -> dict[str, object]:
    profile = QLoRAProfile.load(workspace)
    if max_steps < 1:
        raise PreflightError("max steps must be positive")
    pointer = json.loads(
        (workspace / "artifacts" / "state" / "active_preparation.json").read_text(
            encoding="utf-8"
        )
    )
    processed = Path(str(pointer["processed_path"]))
    if not processed.is_absolute():
        processed = workspace / processed
    processed = processed.resolve()
    train_rows = _rows(processed / "train.jsonl")
    run_input = {
        "qlora_sha256": profile.config_sha256,
        "preparation_fingerprint": pointer["fingerprint"],
        "train_sha256": sha256_file(processed / "train.jsonl"),

    }
    run_sha256 = sha256_bytes(canonical_json(run_input))

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise PreflightError("real QLoRA training requires CUDA")
    snapshot = workspace / "artifacts" / "model-cache" / LOCKED_REVISION
    _verify_model_snapshot(profile, snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    lora = profile.value["lora"]
    peft_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )
    from peft import get_peft_model

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.peft_config["default"].base_model_name_or_path = profile.value["base_model"]["repository"]
    model.peft_config["default"].revision = LOCKED_REVISION

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(profile.training["maximum_sequence_tokens"]),
        )
        encoded["labels"] = list(encoded["input_ids"])
        return cast(dict[str, Any], encoded)

    dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "RUN_INPUT.json").write_bytes(
        canonical_json({**run_input, "run_sha256": run_sha256}) + b"\n"
    )
    arguments = TrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=int(profile.training["micro_batch_size"]),
        gradient_accumulation_steps=int(profile.training["gradient_accumulation_steps"]),
        learning_rate=float(profile.training["learning_rate"]),
        lr_scheduler_type=str(profile.training["scheduler"]),
        warmup_ratio=float(profile.training["warmup_ratio"]),
        optim=str(profile.training["optimizer"]),
        bf16=True,
        gradient_checkpointing=True,
        save_steps=int(profile.training["checkpoint_steps"]),
        logging_steps=1,
        report_to=[],
        seed=int(profile.training["seed"]),
    )
    resume = select_resume_checkpoint(output, run_sha256)

    class BindCheckpointCallback(TrainerCallback):
        def on_save(self, _args: Any, state: Any, _control: Any, **_kwargs: Any) -> None:
            checkpoint = output / f"checkpoint-{int(state.global_step)}"
            if checkpoint.is_dir():
                (checkpoint / "RUN_INPUT.json").write_bytes(
                    canonical_json({"run_sha256": run_sha256}) + b"\n"
                )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        callbacks=[BindCheckpointCallback()],
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    return {"run_sha256": run_sha256, "global_step": trainer.state.global_step}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--acquire-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.acquire_only:
        result = acquire_model(workspace)
    else:
        if args.output is None or args.max_steps is None:
            parser.error("--output and --max-steps are required for training")
        result = train(workspace, args.output.resolve(), args.max_steps)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
