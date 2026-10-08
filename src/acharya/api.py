"""FastAPI app: GET / (web chat for any phone browser / the Android app), GET /health/live,
GET /health/ready, POST /v1/chat (also used by the iOS app)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from acharya.schemas import ChatRequest, ChatResponse, HealthResponse
from acharya.service import ChatService, ModelUnavailable


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(service: ChatService) -> FastAPI:
    app = FastAPI(title="AcharyaGPT", version="1.0.0")
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        history = any("history" in item.get("loc", ()) for item in exc.errors())
        message = ("history must contain complete alternating user/assistant pairs"
                   if history else "request validation failed")
        return error(422, "invalid_request", message)

    @app.exception_handler(ModelUnavailable)
    async def unavailable(_request: Request, exc: ModelUnavailable) -> JSONResponse:
        return error(503, "model_unavailable", f"The model is not available: {exc}")

    def health(live: bool) -> HealthResponse:
        ok, detail = (True, None) if live else service.generator.ready()
        return HealthResponse(status="live" if live else ("ready" if ok else "not_ready"),
                              ready=ok, model=service.generator.name,
                              backend=type(service.generator).__name__,
                              knowledge_base_entries=len(service.index.cards), detail=detail)

    @app.get("/health/live", response_model=HealthResponse)
    def health_live() -> HealthResponse:
        return health(live=True)

    @app.get("/health/ready", response_model=HealthResponse)
    def health_ready() -> JSONResponse:
        response = health(live=False)
        return JSONResponse(status_code=200 if response.ready else 503,
                            content=response.model_dump(mode="json"))

    page = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def web_chat() -> HTMLResponse:
        return HTMLResponse(page)

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:  # sync: runs in FastAPI's threadpool
        return service.chat(request)

    return app
