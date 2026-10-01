from __future__ import annotations

import sys

from acharya.config import Settings
from acharya.ingest.pipeline import build_corpus
from acharya.rag.evaluate import calibrate
from acharya.rag.index import build_index
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


def test_real_pdf_bm25_path_never_imports_dense_stack(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    first = build_corpus(settings)
    second = build_corpus(settings)
    assert first.fingerprint == second.fingerprint
    index = build_index(settings)
    assert calibrate(settings, index).false_support == 0
    response = RAGService(root).chat(
        ChatRequest(message="Which are the doshas involved in Eka kusta?")
    )
    assert response.outcome == "answered"
    assert not {"torch", "transformers", "sentence_transformers"} & set(sys.modules)
