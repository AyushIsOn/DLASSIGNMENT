from __future__ import annotations

import json
from pathlib import Path

import pytest

from acharya.lightning import PreflightError
from training.quality_data import audit_row, compile_reviewed, copy_fraction
from training.quality_eval import concept_match


def row(
    answer: str,
    context: str = "Vata is associated with movement. Kapha is associated with nourishment.",
) -> dict:
    return {
        "id": "example",
        "question": "Explain these traditional concepts.",
        "task_type": "grounded_qa",
        "provenance": {"source": "fixture"},
        "messages": [
            {
                "role": "user",
                "content": (
                    f"CONTEXTS:\n[1] source=fixture page=1\n{context}"
                    "\n\nQUESTION:\nExplain.\n\nANSWER:\n"
                ),
            },
            {"role": "assistant", "content": answer},
        ],
    }


def test_source_copy_rejected_but_original_explanation_not_flagged() -> None:
    source = "Vata is associated with movement. Kapha is associated with nourishment."
    assert copy_fraction(source + " [1]", source) > 0.65
    assert "target_is_mostly_source_copy" in audit_row(row(source + " [1]"))
    assert "target_is_mostly_source_copy" not in audit_row(
        row(
            "These are traditional roles: motion belongs to Vata, "
            "whereas nourishment belongs to Kapha [1]."
        )
    )


def test_ocr_and_historical_extraction_are_flagged() -> None:
    value = row(
        "A source describes traditional movement concepts "
        "rather than establishing a diagnosis [1].",
        "The bod} and V^yu.",
    )
    assert "possible_OCR_or_markup_artifact" in audit_row(value)
    value["task_type"] = "historical_extraction"
    assert "historical_extraction_requires_source_review" in audit_row(value)


def test_unreviewed_data_cannot_be_compiled(tmp_path: Path) -> None:
    benchmark, candidates = tmp_path / "benchmark.jsonl", tmp_path / "candidates.jsonl"
    benchmark.write_text(json.dumps({"query": "Independent question"}) + "\n")
    candidates.write_text(json.dumps({"row": row("Draft"), "split": "train"}) + "\n")
    with pytest.raises(PreflightError, match="reviewer"):
        compile_reviewed(candidates, tmp_path / "quality", benchmark)
    assert not (tmp_path / "quality").exists()


def test_concept_screening_requires_every_group() -> None:
    assert concept_match("Vata concerns movement.", [["vata"], ["movement", "motion"]])
    assert not concept_match("Vata is a dosha.", [["vata"], ["movement", "motion"]])


def test_reviewed_source_cannot_leak_between_splits(tmp_path: Path) -> None:
    benchmark, candidates = tmp_path / "benchmark.jsonl", tmp_path / "candidates.jsonl"
    benchmark.write_text(json.dumps({"query": "Independent question"}) + "\n")
    rows = []
    for split in ("train", "test"):
        value = row(
            "These traditional roles describe motion for Vata "
            "and supportive nourishment for Kapha [1]."
        )
        value["id"] = split
        value["question"] = f"Explain the concepts in the {split} example."
        rows.append(
            {
                "row": value,
                "split": split,
                "reviewer": "fixture reviewer",
                "review_status": "approved",
                "review_checks": {
                    "factuality": True,
                    "relevance": True,
                    "clarity": True,
                    "safety": True,
                },
            }
        )
    candidates.write_text("".join(json.dumps(value) + "\n" for value in rows))
    with pytest.raises(PreflightError, match="leakage"):
        compile_reviewed(candidates, tmp_path / "quality", benchmark)


def test_independent_references_are_not_passed_to_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acharya.schemas import ChatResponse
    from training.quality_eval import evaluate

    seen = []

    class Service:
        def __init__(self, workspace: Path) -> None:
            pass

        def chat(self, request):
            seen.append(request.message)
            return ChatResponse(answer="No evidence.", outcome="abstained", citations=(), scores=())

    monkeypatch.setattr("training.quality_eval.RAGService", Service)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "id": "one",
                "query": "A fresh question?",
                "expected_outcome": "abstained",
                "concept_groups": [["SECRET_REFERENCE"]],
                "review_notes": "SECRET_REFERENCE",
            }
        )
        + "\n"
    )
    report = evaluate(tmp_path, benchmark, None, tmp_path / "results", "cpu")
    assert seen == ["A fresh question?"]
    assert report["answer_quality_established"] is False


def test_legacy_training_is_blocked_before_model_loading(workspace_factory, tmp_path: Path) -> None:
    from training.train_qlora import train

    workspace = workspace_factory()
    state = workspace / "artifacts/state"
    state.mkdir(parents=True)
    (state / "active_preparation.json").write_text(
        json.dumps({"processed_path": "data/processed/legacy", "fingerprint": "fixture"})
    )
    with pytest.raises(PreflightError, match="Legacy copy-target"):
        train(workspace, tmp_path / "new-training", 10)


def test_rewrite_produces_pending_drafts_not_approved_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.rewrite_targets import rewrite

    state = tmp_path / "artifacts/state"
    state.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    (state / "active_preparation.json").write_text(
        json.dumps({"processed_path": "data", "fingerprint": "fixture"})
    )
    original = row("Vata is associated with movement. Kapha is associated with nourishment. [1]")
    original["provenance"] = {"dataset": "fixture"}
    original["messages"].insert(0, {"role": "system", "content": "Use only the context."})
    (data / "train.jsonl").write_text(json.dumps(original) + "\n")
    for split in ("validation", "test"):
        (data / f"{split}.jsonl").write_text("")
    seen = []

    def teacher(workspace, adapter):
        assert adapter is None

        def generate(prompt, tokens):
            seen.append(prompt)
            return json.dumps(
                {
                    "question": "How do these traditional principles differ?",
                    "answer": "These traditional roles describe motion for Vata "
                    "and supportive nourishment for Kapha [1].",
                }
            )

        return generate

    monkeypatch.setattr("training.rewrite_targets.make_generator", teacher)
    output = tmp_path / "drafts.jsonl"
    rewrite(tmp_path, output, 1, 1)
    candidate = json.loads(output.read_text())
    assert candidate["review_status"] == "pending"
    assert candidate["reviewer"] is None
    assert candidate["row"]["task_type"] == "explanatory_qa"
    assert len(seen) == 1
    rewrite(tmp_path, output, 1, 1)
    assert len(seen) == 1
