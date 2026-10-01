from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from acharya.api import create_app


def test_health_chat_and_validation_contracts(built_workspace: Path) -> None:
    client = TestClient(create_app(built_workspace))
    assert client.get("/health/live").json()["status"] == "live"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["retrieval_mode"] == "bm25_only"
    answer = client.post(
        "/v1/chat", json={"message": "Which are the doshas involved in Eka kusta?"}
    )
    assert answer.status_code == 200
    body = answer.json()
    assert body["outcome"] == "answered"
    assert body["citations"]
    assert body["scores"][0]["dense"] is None
    invalid = client.post("/v1/chat", json={"message": "   "})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
    malformed_histories = (
        [{"role": "user", "content": "odd-sensitive-value"}],
        [
            {"role": "assistant", "content": "first-sensitive-value"},
            {"role": "user", "content": "second-sensitive-value"},
        ],
        [
            {"role": "user", "content": "one-sensitive-value"},
            {"role": "user", "content": "two-sensitive-value"},
        ],
        [
            {"role": "user", "content": "one-sensitive-value"},
            {"role": "assistant", "content": "two-sensitive-value"},
            {"role": "assistant", "content": "three-sensitive-value"},
            {"role": "user", "content": "four-sensitive-value"},
        ],
    )
    for history in malformed_histories:
        response = client.post("/v1/chat", json={"message": "hello", "history": history})
        assert response.status_code == 422
        body = response.json()["error"]
        assert body == {
            "code": "invalid_request",
            "message": "history must contain complete alternating user/assistant pairs",
        }
        assert "sensitive-value" not in response.text
