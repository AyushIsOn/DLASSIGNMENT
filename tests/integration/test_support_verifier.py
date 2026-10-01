from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.rag.evaluate import calibrate_support, load_support_calibration


class CalibrationScorer:
    def score(self, _premises: tuple[str, ...], hypothesis: str) -> float:
        unsupported = hypothesis.startswith(
            (
                "Vata is not",
                "Eka kusta is associated only",
                "Ayurveda describes seven",
                "Weak agni",
                "These characteristics",
                "Every adult",
                "Ayurveda has been",
                "Diesel",
            )
        )
        return 0.2 if unsupported else 0.95


def test_support_calibration_zero_false_accept_and_hash_binding(project_root: Path) -> None:
    settings = Settings.load(project_root)
    result = calibrate_support(settings, CalibrationScorer())
    assert result.ready
    assert result.false_accepts == 0
    assert result.threshold == 0.95
    assert load_support_calibration(settings) == result
