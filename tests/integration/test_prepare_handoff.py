from __future__ import annotations

import csv
from pathlib import Path

from acharya.config import Settings
from acharya.ingest.download import Acquisition
from training.prepare_sft import prepare, verify_preparation


def _write_rows(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_merged_preparation_is_reproducible_and_reconciled(workspace_factory: object) -> None:
    root = workspace_factory()  # type: ignore[operator]
    settings = Settings.load(root)
    acquisitions = []
    for dataset_id, spec in settings.datasets["kaggle"]["datasets"].items():
        spec["expected_rows"] = 12
        files = root / "fixtures" / spec["slug"]
        path = files / spec["primary_member"]
        rows = []
        for index in range(12):
            row = {header: f"field {index}" for header in spec["headers"]}
            if dataset_id == "gpreda/medquad/1":
                row.update(
                    {
                        "question": f"What is educational concept number {index}?",
                        "answer": (
                            f"Educational background number {index} describes general health "
                            "concepts without individual advice."
                        ),
                        "source": f"Source {index}",
                        "focus_area": "education",
                    }
                )
            elif dataset_id == "akashkumarpr/ayurvedic-knowledge-dataset/1":
                row.update(
                    {
                        "Ayurvedic Name": f"Concept {index}",
                        "Modern Equivalent": (
                            f"Educational background description number {index} with "
                            "sufficiently detailed source context."
                        ),
                        "Source Text": f"Classical source {index}",
                    }
                )
            else:
                row.update(
                    {
                        "Problem": f"General concept {index}",
                        "Symptoms": (
                            f"Educational contextual description number {index} with "
                            "sufficient background detail."
                        ),
                        "Source": f"Published source {index}",
                        "Confidence": "0.75",
                        "Authentication Notes": f"Reviewed with cited source {index}",
                        "Auth_Score": "0.90",
                    }
                )
            rows.append(row)
        _write_rows(path, spec["headers"], rows)
        acquisitions.append(
            Acquisition(
                dataset_id,
                files / "archive.zip",
                files,
                str(spec["archive_sha256"]),
                {str(spec["primary_member"]): "fixture"},
                False,
                {"fixture": True},
            )
        )
    first = prepare(settings, tuple(acquisitions))
    second = prepare(settings, tuple(acquisitions))
    assert first.fingerprint == second.fingerprint
    assert first.hashes == second.hashes
    assert all(first.split_counts[name] > 0 for name in ("train", "validation", "test"))
    assert all(
        counts["input"]
        == counts["accepted_unique"]
        + counts["exact_dropped"]
        + counts["near_dropped"]
        + counts["rejected"]
        for counts in first.source_counts.values()
    )
    assert verify_preparation(settings, strict=True)["verified"] is True
    healthcare = first.source_counts["aliainaanraza/ayurveda-healthcare-dataset/2"]
    assert healthcare["retrieval_chunks"] == 0
