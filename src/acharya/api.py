"""FastAPI application for the offline Gate A service."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from acharya.providers.runtime import configured_provider
from acharya.rag.service import RAGService, ServiceUnavailable
from acharya.schemas import ChatRequest, ChatResponse, HealthResponse


def create_app(
    workspace: Path | str | None = None, *, service: RAGService | None = None
) -> FastAPI:
    root = workspace or os.environ.get("ACHARYA_WORKSPACE") or Path(__file__).resolve().parents[2]
    if service is None:
        provider = configured_provider(Path(root), os.environ)
        service = RAGService(
            root,
            generative_provider=provider,
            generation_mode="optional" if provider is not None else "extractive",
        )
    app = FastAPI(title="AcharyaGPT", version="0.1.0")
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        invalid_history = any("history" in item.get("loc", ()) for item in error.errors())
        message = (
            "history must contain complete alternating user/assistant pairs"
            if invalid_history
            else "request validation failed"
        )
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": message}},
        )

    @app.exception_handler(ServiceUnavailable)
    async def unavailable(_request: Request, _error: ServiceUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "retrieval_not_ready",
                    "message": "calibrated local retrieval is unavailable",
                }
            },
        )

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return service.health(live_only=True)

    @app.get("/health/ready", response_model=HealthResponse)
    async def health_ready() -> HealthResponse:
        response = service.health()
        if not response.ready:
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))  # type: ignore[return-value]
        return response

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return service.chat(request)

    return app


def app_factory() -> FastAPI:
    return create_app()


app = app_factory()
