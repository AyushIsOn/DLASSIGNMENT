from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.ingest.loaders import SourceRecord
from acharya.ingest.normalize import make_chunks, normalize_text
from acharya.safety import SafetyPolicy


def test_normalize_repairs_layout_spacing() -> None:
    assert normalize_text(" Vata   Dosha \n is here .") == "Vata Dosha is here."


def test_chunk_ids_and_roles_are_deterministic(project_root: Path) -> None:
    settings = Settings.load(project_root)
    policy = SafetyPolicy(settings.safety)
    records = (
        SourceRecord("fixture", 1, 1, "What is Vata?", "Vata is an educational concept."),
        SourceRecord("fixture", 1, 2, "Treatment?", "Avoid curd as a treatment."),
    )
    first = make_chunks(records, policy, 20, 2)
    second = make_chunks(records, policy, 20, 2)
    assert first == second
    assert first[0].role == "educational"
    assert first[1].role == "treatment"
