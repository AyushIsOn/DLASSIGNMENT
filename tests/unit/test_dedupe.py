from __future__ import annotations

from acharya.ingest.dedupe import deduplicate, jaccard, minhash_signature, shingles
from acharya.ingest.normalize import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, "fixture", 1, 1, 1, "question", text, "educational", "p")


def test_exact_duplicate_winner_uses_canonical_id() -> None:
    chunks = (_chunk("chk_b", "same exact text"), _chunk("chk_a", "same exact text"))
    winners, components = deduplicate(chunks)
    assert [item.chunk_id for item in winners] == ["chk_a"]
    assert components[0].winner_id == "chk_a"
    assert components[0].kind == "exact"


def test_similarity_and_minhash_are_deterministic() -> None:
    values = shingles("one two three four five six")
    assert jaccard(values, values) == 1.0
    assert minhash_signature(values) == minhash_signature(values)
