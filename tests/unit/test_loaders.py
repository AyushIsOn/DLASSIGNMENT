from __future__ import annotations

import csv
from pathlib import Path

import pytest

from acharya.config import ConfigurationError, Settings
from acharya.ingest.loaders import (
    extract_pdf_pages,
    header_sha256,
    load_csv_dataset,
    parse_qa_pages,
)


def test_real_pdf_extracts_six_pages_and_stable_records(project_root: Path) -> None:
    settings = Settings.load(project_root)
    pdf = settings.ingestion["pdf"]
    pages = extract_pdf_pages(settings.pdf_path, pdf["sha256"], pdf["expected_pages"])
    assert len(pages) == 6
    records = parse_qa_pages(pages)
    assert len(records) == 19
    assert records[0].page == 1
    assert "scaly" in records[0].question
    assert len({item.provenance for item in records}) == len(records)


def test_exact_header_order_and_healthcare_decimal_rules(
    tmp_path: Path, project_root: Path
) -> None:
    spec = Settings.load(project_root).datasets["kaggle"]["datasets"][
        "aliainaanraza/ayurveda-healthcare-dataset/2"
    ]
    path = tmp_path / "health.csv"
    row = {name: "supporting value" for name in spec["headers"]}
    row.update(
        {
            "ID": "1",
            "Problem": "Seasonal wellbeing background",
            "Symptoms": "General educational symptoms and background context.",
            "Source": "Classical source",
            "Confidence": "0.60",
            "Authentication Notes": "Reviewed against the cited source",
            "Auth_Score": "0.80",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec["headers"])
        writer.writeheader()
        writer.writerow(row)
    loaded = load_csv_dataset(
        path,
        "aliainaanraza/ayurveda-healthcare-dataset/2",
        spec,
        enforce_row_count=False,
    )
    assert len(loaded.records) == 1
    assert loaded.header_sha256 == spec["header_sha256"] == header_sha256(spec["headers"])
    reordered = tmp_path / "reordered.csv"
    with reordered.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reversed(spec["headers"])))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ConfigurationError, match="schema_header_mismatch"):
        load_csv_dataset(
            reordered,
            "aliainaanraza/ayurveda-healthcare-dataset/2",
            spec,
            enforce_row_count=False,
        )
    row["Auth_Score"] = "NaN"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spec["headers"])
        writer.writeheader()
        writer.writerow(row)
    rejected = load_csv_dataset(
        path,
        "aliainaanraza/ayurveda-healthcare-dataset/2",
        spec,
        enforce_row_count=False,
    )
    assert rejected.rejected[0].reason == "invalid_auth_score"
