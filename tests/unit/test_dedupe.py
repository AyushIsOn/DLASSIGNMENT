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


def test_prefix_filter_matches_brute_force_components() -> None:
    import random

    rng = random.Random(42)
    chunks = tuple(
        _chunk(f"{i:03}", " ".join(rng.sample([f"word{n}" for n in range(25)], rng.randint(1, 20))))
        for i in range(100)
    )
    for threshold in (0.3, 0.6, 0.9, 1.0):
        groups = [{i} for i in range(len(chunks))]
        for i, left in enumerate(chunks):
            for j in range(i):
                if jaccard(shingles(left.text, 1), shingles(chunks[j].text, 1)) >= threshold:
                    merged = groups[i] | groups[j]
                    for member in merged:
                        groups[member] = merged
        expected = {min(group) for group in groups}
        winners, _ = deduplicate(chunks, threshold, 1)
        assert {int(item.chunk_id) for item in winners} == expected
