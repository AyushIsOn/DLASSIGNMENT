"""CPU-only library/tokenizer integration smoke; never loads the 8B weights."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from acharya.config import canonical_json
from acharya.lightning import QLoRAProfile
from training.train_qlora import _rows, _tokenize_chat_row, training_arguments


def check(workspace: Path, tokenizer_path: Path) -> dict[str, object]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Qwen3Config,
        Qwen3ForCausalLM,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(3407)
    torch.set_num_threads(2)
    profile = QLoRAProfile.load(workspace)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    pointer = json.loads((workspace / "artifacts/state/active_preparation.json").read_text())
    processed = workspace / pointer["processed_path"]
    evidence: dict[str, object] = {
        "fingerprint": pointer["fingerprint"],
        "actual_8b_training": False,
    }
    encoded_rows = []
    for split in ("train", "validation", "test"):
        encoded = [
            _tokenize_chat_row(row, tokenizer, int(profile.training["maximum_sequence_tokens"]))
            for row in _rows(processed / f"{split}.jsonl")
        ]
        lengths = sorted(len(row["input_ids"]) for row in encoded)
        evidence[split] = {
            "rows": len(encoded),
            "min_tokens": lengths[0],
            "max_tokens": lengths[-1],
            "p95_tokens": lengths[int(len(lengths) * 0.95)],
            "tokens": sum(lengths),
            "truncated": 0,
            "supervised_tokens": sum(sum(x != -100 for x in row["labels"]) for row in encoded),
        }
        if split == "train":
            encoded_rows = sorted(encoded, key=lambda row: len(row["input_ids"]))[:4]
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    batch = collator(encoded_rows)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            head_dim=8,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    with tempfile.TemporaryDirectory(prefix="acharya-contract-") as temporary:
        options = training_arguments(Path(temporary), profile, 2)
        options.update(
            use_cpu=True,
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=False,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            disable_tqdm=True,
        )
        dataset = Dataset.from_list(encoded_rows)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**options),
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        result = trainer.train()
        metrics = trainer.evaluate()
        assert result.global_step == 2 and torch.isfinite(torch.tensor(metrics["eval_loss"]))
        evidence["tiny_cpu_trainer"] = {
            "steps": result.global_step,
            "eval_loss": metrics["eval_loss"],
            "variable_length_padding": True,
            "model": "random 1-layer Qwen3 + LoRA",
        }
        # Resume the same immutable two-step schedule from step one in a new folder.
        import shutil

        from peft import get_peft_model_state_dict

        from acharya.checkpoints import pack, restore, seal

        expected = {
            key: tensor.detach().clone() for key, tensor in get_peft_model_state_dict(model).items()
        }
        first = Path(temporary) / "checkpoint-1"
        (first / "RUN_INPUT.json").write_text('{"run_sha256":"cpu-contract"}')
        seal(first)
        archive = Path(temporary) / "recovery.tar"
        pack(workspace, first, archive)
        transfer = Path(temporary) / "other-account"
        (transfer / "configs").mkdir(parents=True)
        shutil.copy2(workspace / "configs/qlora.yaml", transfer / "configs/qlora.yaml")
        (transfer / "artifacts/state").mkdir(parents=True)
        shutil.copy2(
            workspace / "artifacts/state/active_preparation.json",
            transfer / "artifacts/state/active_preparation.json",
        )
        restored = restore(transfer, archive)
        torch.manual_seed(3407)
        fresh = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=2048,
                head_dim=8,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
        fresh = get_peft_model(
            fresh,
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ),
        )
        options["output_dir"] = str(transfer / "artifacts/external-run/checkpoint")
        resumed = Trainer(
            model=fresh,
            args=TrainingArguments(**options),
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        resumed.train(resume_from_checkpoint=str(restored["checkpoint"]))
        actual = get_peft_model_state_dict(fresh)
        maximum_difference = max(
            float((expected[key] - actual[key]).abs().max()) for key in expected
        )
        assert resumed.state.global_step == 2 and maximum_difference < 1e-7
        evidence["portable_cpu_resume"] = {
            "from_step": 1,
            "to_step": 2,
            "optimizer_scheduler_rng_restored": True,
            "maximum_parameter_difference": maximum_difference,
        }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.workspace.resolve(), args.tokenizer.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
