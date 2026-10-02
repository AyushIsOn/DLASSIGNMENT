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
