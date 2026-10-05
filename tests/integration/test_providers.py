from __future__ import annotations

from pathlib import Path

from acharya.config import Settings
from acharya.providers.base import ProviderRequest, ProviderResult
from acharya.rag.evaluate import calibrate_support
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest


class AlwaysSupported:
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
                "Unsupported",
            )
        )
        return 0.1 if unsupported else 0.95


class FakeProvider:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        return ProviderResult(self.text, (), self.name)


def test_generated_candidate_is_gated_and_rejected_text_never_leaks(built_workspace: Path) -> None:
    settings = Settings.load(built_workspace)
    source_eval = Path(__file__).resolve().parents[2] / "eval" / "support_verifier.jsonl"
    (built_workspace / "eval").mkdir(exist_ok=True)
    (built_workspace / "eval" / "support_verifier.jsonl").write_bytes(source_eval.read_bytes())
    (built_workspace / "prompts").mkdir(exist_ok=True)
    source_prompt = Path(__file__).resolve().parents[2] / "prompts" / "grounded_v1.txt"
    (built_workspace / "prompts" / "grounded_v1.txt").write_bytes(source_prompt.read_bytes())
    calibrate_support(settings, AlwaysSupported())
    accepted = FakeProvider("Grounded statement [1].")
    service = RAGService(
        built_workspace,
        generative_provider=accepted,
        generation_mode="optional",
        support_scorer=AlwaysSupported(),
    )
    response = service.chat(ChatRequest(message="Which are the doshas involved in Eka kusta?"))
    assert response.answer == "Grounded statement [1]."
    assert len(accepted.requests) == 1

    rejected = FakeProvider("Unsupported fabricated candidate [1].")
    fallback = RAGService(
        built_workspace,
        generative_provider=rejected,
        generation_mode="optional",
        support_scorer=AlwaysSupported(),
    ).chat(ChatRequest(message="Which are the doshas involved in Eka kusta?"))
    assert "Unsupported fabricated" not in fallback.answer
    assert fallback.citations
    assert len(rejected.requests) == 2
