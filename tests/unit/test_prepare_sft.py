from __future__ import annotations

from acharya.ingest.normalize import Chunk
from training.prepare_sft import _sft_row, _split


def test_sft_rows_and_group_splits_are_deterministic() -> None:
    chunk = Chunk(
        "chk_fixed",
        "fixture",
        0,
        1,
        1,
        "What is this?",
        "Educational background text.",
        "educational",
        "fixture:row=1",
        1,
        "row=1",
        "CC-BY-4.0",
        "https://example.invalid",
        "secondary",
        "Source",
    )
    assert _split(chunk) == _split(chunk)
    row = _sft_row(chunk)
    assert row["messages"][0] == {"role": "user", "content": "What is this?"}  # type: ignore[index]
    assert row["provenance"]["locator"] == "row=1"  # type: ignore[index]


def test_chunks_of_one_record_stay_in_same_split() -> None:
    from dataclasses import replace

    chunk = Chunk("a", "fixture", 1, 1, 1, "q", "text", "educational", "source:row=1")
    assert len({_split(replace(chunk, chunk_id=f"chunk_{i}")) for i in range(100)}) == 1


def test_repeated_question_across_sources_stays_in_one_split() -> None:
    from dataclasses import replace

    chunk = Chunk("a", "first", 1, 1, 1, "What is Vata?", "text", "educational", "row=1")
    other = replace(chunk, chunk_id="b", source="second", provenance="row=99",
                    question="WHAT is  Vata !", text="different answer")
    assert _split(chunk) == _split(other)


def test_sft_preserves_a_complete_long_answer(project_root: object) -> None:
    from acharya.config import Settings
    from acharya.ingest.loaders import SourceRecord
    from acharya.safety import SafetyPolicy
    from training.prepare_sft import _complete_sft

    settings = Settings.load(project_root)  # type: ignore[arg-type]
    answer = " ".join(f"educational{i}" for i in range(400))
    record = SourceRecord("fixture", 1, 1, "What is this?", answer, role_hint="educational")
    rows, _ = _complete_sft([record], SafetyPolicy(settings.safety), {"fixture"}, 0.9, 5)
    assert len(rows) == 1
    assert rows[0].text == answer
