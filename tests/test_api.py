from fastapi.testclient import TestClient

from acharya.api import create_app
from acharya.generation import EchoGenerator, GenerationError
from acharya.prompting import SYSTEM_PROMPT
from acharya.retrieval import BM25, Card, tokenize
from acharya.service import ChatService


def client(answer: str = "Vata, Pitta and Kapha are the three doshas.") -> tuple[TestClient,
                                                                                 EchoGenerator]:
    generator = EchoGenerator(answer)
    return TestClient(create_app(ChatService(generator))), generator


def test_health_and_chat_with_retrieval():
    api, generator = client()
    assert api.get("/health/live").json()["status"] == "live"
    ready = api.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["knowledge_base_entries"] > 1000
    body = api.post("/v1/chat", json={"message": "What is the modern equivalent of Amlapitta?",
                                      "history": []}).json()
    assert body["outcome"] == "answered" and body["mode"] == "finetuned_rag"
    assert body["citations"] and any("amlapitta" in c["text"].casefold() for c in body["citations"])
    prompt = generator.prompts[-1]
    assert prompt.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")
    assert "Knowledge base entries:\n[1] " in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_small_talk_is_closed_book():
    api, generator = client("Namaste!")
    body = api.post("/v1/chat", json={"message": "Hello", "history": []}).json()
    assert body["mode"] == "finetuned" and body["citations"] == []
    assert "Knowledge base entries" not in generator.prompts[-1]


def test_follow_up_uses_history_for_retrieval_and_prompt():
    api, generator = client("It is predominantly a Pitta disorder.")
    history = [{"role": "user", "content": "What is the modern equivalent of Amlapitta?"},
               {"role": "assistant", "content": "Amlapitta corresponds to hyperacidity."}]
    body = api.post("/v1/chat", json={"message": "Which dosha is involved in it?",
                                      "history": history}).json()
    assert body["outcome"] == "answered"
    assert "<|im_start|>assistant\nAmlapitta corresponds to hyperacidity.<|im_end|>" in \
        generator.prompts[-1]
    assert any("Amlapitta" in c["text"] for c in body["citations"])


def test_safety_routing_never_calls_the_model():
    api, generator = client()
    for message, outcome in [("I have severe chest pain spreading to my arm", "urgent"),
                             ("I want to end my life", "urgent"),
                             ("How much Ashwagandha should I take daily?", "refused"),
                             ("What is the right dose of triphala for me?", "refused")]:
        body = api.post("/v1/chat", json={"message": message, "history": []}).json()
        assert body["outcome"] == outcome and body["mode"] == "safety", message
    assert generator.prompts == []


def test_educational_questions_are_not_refused():
    api, _ = client()
    for message in ["What is Panchakarma?", "Which herbs are used for asthma in Ayurveda?",
                    "What are the symptoms of Kamala?", "How is Ardita treated in Ayurveda?"]:
        assert api.post("/v1/chat", json={"message": message}).json()["outcome"] == "answered"


def test_dose_in_generated_answer_is_replaced():
    api, _ = client("Take 5 g of Ashwagandha twice daily.")
    body = api.post("/v1/chat", json={"message": "Tell me about Ashwagandha"}).json()
    assert body["outcome"] == "refused" and body["warning"] == "dose_removed"
    assert "5 g" not in body["answer"]


def test_invalid_history_is_422_with_error_envelope():
    api, _ = client()
    response = api.post("/v1/chat", json={"message": "hi",
                                          "history": [{"role": "user", "content": "x"}]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert api.post("/v1/chat", json={"message": "   "}).status_code == 422


def test_backend_failure_is_503():
    class Broken(EchoGenerator):
        def generate(self, prompt: str, max_new_tokens: int) -> str:
            raise GenerationError("ollama is not running")

    api = TestClient(create_app(ChatService(Broken())))
    response = api.post("/v1/chat", json={"message": "What is Vata?"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_bm25_ranks_exact_entity_first():
    index = BM25([Card("a", "Amlapitta", "s", "Amlapitta hyperacidity Pitta"),
                  Card("b", "Ardita", "s", "Ardita facial paralysis Vata")])
    assert index.search("What is Ardita?", k=1)[0].card.id == "b"
    assert index.search("hello there", k=3) == []
    assert tokenize("What are the symptoms?") == ["symptom"]


def test_web_chat_page_is_served():
    from fastapi.testclient import TestClient

    from acharya.api import create_app
    from acharya.generation import EchoGenerator
    from acharya.service import ChatService

    client = TestClient(create_app(ChatService(EchoGenerator())))
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/v1/chat" in page.text and "AcharyaGPT" in page.text
