import json
from pathlib import Path

import pytest

from acharya.prompting import (
    SYSTEM_PROMPT,
    build_messages,
    clean_generation,
    completion_text,
    render_prompt,
)
from finetune.data import IGNORE_INDEX, Collator, encode, prompt_text, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "artifacts/models/Qwen3-8B"


def test_render_matches_qwen3_chat_template_format():
    messages = build_messages("Q2", history=[{"role": "user", "content": "Q1"},
                                             {"role": "assistant", "content": "A1"}])
    assert render_prompt(messages) == (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_open_book_prompt_numbers_entries():
    content = build_messages("Which dosha?", ["entry one", "entry two"])[-1]["content"]
    assert content == ("Knowledge base entries:\n[1] entry one\n[2] entry two\n\n"
                       "Question: Which dosha?")


def test_render_rejects_bad_order():
    with pytest.raises(ValueError):
        render_prompt([{"role": "system", "content": "s"}])


def test_clean_generation_strips_stop_tokens_and_think_blocks():
    assert clean_generation("Answer.<|im_end|>\n") == "Answer."
    assert clean_generation("<think>\nhidden\n</think>\n\nVisible") == "Visible"
    assert completion_text(" Vata. ") == "Vata.<|im_end|>"


def test_dataset_matches_manifest_and_prompt():
    verify_manifest(ROOT / "data/sft", ROOT / "data/kb/cards.jsonl")


def test_no_test_question_or_heldout_entity_in_training():
    train = [json.loads(line) for line in (ROOT / "data/sft/train.jsonl").open()]
    test = [json.loads(line) for line in (ROOT / "data/sft/test.jsonl").open()]
    train_questions = {row["question"].casefold() for row in train}
    trained_entities = {row["meta"]["entity"] for row in train}
    for row in test:
        if row["meta"]["group"] in {"seen_closed", "concepts", "heldout_open", "heldout_closed"}:
            assert row["question"].casefold() not in train_questions, row["id"]
        if row["meta"]["group"].startswith("heldout"):
            assert row["meta"]["entity"] not in trained_entities, row["id"]
    # the old failure mode: answers that just copy the prompt
    copies = sum(row["messages"][-1]["content"] in row["messages"][-2]["content"] for row in train)
    assert copies / len(train) < 0.01


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").is_file(), reason="tokenizer not downloaded")
def test_label_mask_and_chat_template_with_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    row = json.loads((ROOT / "data/sft/validation.jsonl").open().readline())
    official = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, *row["messages"][:-1]],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert official == prompt_text(row)
    item = encode(row, tokenizer, 1024)
    supervised = [t for t, label in zip(item["input_ids"], item["labels"], strict=True)
                  if label != IGNORE_INDEX]
    assert tokenizer.decode(supervised) == completion_text(row["messages"][-1]["content"])
    batch = Collator(tokenizer.pad_token_id)([item, encode(row | {"messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        tokenizer, 1024)])
    assert batch["input_ids"].shape[1] % 8 == 0
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
