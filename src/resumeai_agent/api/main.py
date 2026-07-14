"""FastAPI entry point for ResumeAI-Agent model inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request

from resumeai_agent.api.schemas import HealthResponse, MatchRequest, MatchResponse
from resumeai_agent.services.matching import MODEL_NAME, MatchingService


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "tfidf_logistic_regression.joblib"


def create_app(service: MatchingService | None = None) -> FastAPI:
    """Create an app with an injectable service for isolated API tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.matching_service = service or MatchingService.from_artifact(DEFAULT_MODEL_PATH)
        yield

    app = FastAPI(
        title="ResumeAI-Agent API",
        version="0.1.0",
        description="Classifies the fit between an English resume and job description.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_name=MODEL_NAME)

    @app.post("/api/v1/match", response_model=MatchResponse, tags=["matching"])
    def match(request: Request, payload: MatchRequest) -> MatchResponse:
        result = request.app.state.matching_service.predict(
            payload.resume_text, payload.job_description_text
        )
        return MatchResponse(**result)

    return app


app = create_app()
