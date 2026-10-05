from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.rag.index import build_index, load_index, tokenize


def test_index_is_deterministic_and_raw(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    first = build_index(settings)
    second = build_index(settings)
    assert first.fingerprint == second.fingerprint
    assert tuple(item.chunk_id for item in first.documents) == tuple(
        sorted(item.chunk_id for item in first.documents)
    )
    assert load_index(settings).fingerprint == first.fingerprint
    assert first.score(tokenize("Eka kusta dosha"), 0) != 0.75


def test_dense_input_uses_existing_source_question_and_text(built_workspace: Path) -> None:
    from acharya.rag.index import retrieval_passage

    class Encoder:
        def __init__(self) -> None:
            self.passages: list[str] = []

        def encode_passages(self, passages: list[str]) -> list[list[float]]:
            self.passages = passages
            return [[1.0, 0.0] for _ in passages]

        def encode_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    encoder = Encoder()
    index = build_index(Settings.load(built_workspace), "mvp_hybrid", embedder=encoder)
    assert encoder.passages == [retrieval_passage(document) for document in index.documents]
    assert all(
        document.text in passage
        for document, passage in zip(index.documents, encoder.passages, strict=True)
    )
    assert "passage_format" in (index.model_hashes or {})
