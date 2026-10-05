from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from acharya.lightning import LOCKED_REVISION, PreflightError, QLoRAProfile, cpu_dry_run
from training.train_qlora import _tokenize_chat_row


def test_locked_qlora_profile_and_cpu_dry_run(project_root: Path) -> None:
    profile = QLoRAProfile.load(project_root)
    assert profile.value["base_model"]["revision"] == LOCKED_REVISION
    assert profile.value["quantization"] == {
        "load_in_4bit": False,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": False,
    }
    assert profile.value["training"]["effective_batch_size"] == 16
    result = cpu_dry_run()
    assert result["steps"] == 5
    assert result["saved_checkpoint"] is False
    assert result["losses"] == sorted(result["losses"], reverse=True)


def test_profile_rejects_full_tuning_or_relaxed_caps(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    path = root / "configs" / "qlora.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["method"] = "full"
    value["external"]["maximum_wall_minutes"] = 481
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(PreflightError):
        QLoRAProfile.load(root)


def test_training_masks_prompt_tokens() -> None:
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool = False,
            enable_thinking: bool = False,
        ) -> str:
            rendered = " ".join(f"{item['role']}:{item['content']}" for item in messages)
            return rendered + (" assistant:" if add_generation_prompt else "")

        def __call__(
            self, text: str, *, add_special_tokens: bool, **_: object
        ) -> dict[str, list[int]]:
            return {"input_ids": [ord(char) for char in text]}

    result = _tokenize_chat_row(
        {
            "messages": [
                {"role": "user", "content": "What is Vata?"},
                {"role": "assistant", "content": "A traditional Ayurvedic concept."},
            ]
        },
        FakeTokenizer(),
        128,
    )
    assert result["labels"][0] == -100
    assert any(label != -100 for label in result["labels"])
    with pytest.raises(PreflightError, match="exceeds maximum"):
        _tokenize_chat_row(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "long answer"},
                ]
            },
            FakeTokenizer(),
            20,
        )


def test_training_rejects_incompatible_prefix() -> None:
    class IncompatibleTokenizer:
        def apply_chat_template(self, *_: object, **kwargs: object) -> str:
            return "different" if kwargs.get("add_generation_prompt") else "answer"

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [ord(char) for char in text]}

    with pytest.raises(PreflightError, match="prefix does not match"):
        _tokenize_chat_row(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            IncompatibleTokenizer(),
            128,
        )
