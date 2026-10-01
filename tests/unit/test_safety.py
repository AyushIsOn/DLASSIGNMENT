from __future__ import annotations

from pathlib import Path

from acharya.safety import SafetyAction, SafetyPolicy, normalize_safety_text


def _policy(project_root: Path) -> SafetyPolicy:
    return SafetyPolicy.load(project_root / "configs" / "safety.yaml")


def test_precedence_and_positive_cases(project_root: Path) -> None:
    policy = _policy(project_root)
    assert policy.classify_query("Diagnose me; I cannot breathe.").action is SafetyAction.URGENT
    assert policy.classify_query("I'm suicidal").action is SafetyAction.SELF_HARM
    assert policy.classify_query("Do I have psoriasis?").action is SafetyAction.DIAGNOSIS
    assert policy.classify_query("What should I take?").action is SafetyAction.INDIVIDUAL_TREATMENT
    assert policy.classify_query("What dosage is right?").action is SafetyAction.DOSE


def test_benign_educational_query_retrieves(project_root: Path) -> None:
    decision = _policy(project_root).classify_query("Which doshas are described for Eka kusta?")
    assert decision.action is SafetyAction.RETRIEVE
    assert decision.response is None


def test_candidate_is_rejected_whole(project_root: Path) -> None:
    policy = _policy(project_root)
    assert not policy.candidate_allowed("General context. You should take 5 mg twice a day.")
    assert policy.candidate_allowed("Vata and Kapha are described in the source.")


def test_unicode_normalization() -> None:
    assert normalize_safety_text("\uff24\uff2f  I\u00a0HAVE") == "do i have"
