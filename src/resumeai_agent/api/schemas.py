"""Request and response contracts for the model API."""

from typing import Literal

from pydantic import BaseModel, Field


MatchLabel = Literal["No Fit", "Potential Fit", "Good Fit"]


class MatchRequest(BaseModel):
    """A candidate resume and a job description to compare."""

    resume_text: str = Field(min_length=1, max_length=50_000)
    job_description_text: str = Field(min_length=1, max_length=50_000)


class MatchResponse(BaseModel):
    """Structured classification result from the deployed baseline model."""

    label: MatchLabel
    probabilities: dict[MatchLabel, float]
    model_name: str


class HealthResponse(BaseModel):
    """Readiness result for monitoring and deployment checks."""

    status: Literal["ok"]
    model_name: str


class AnalysisReport(BaseModel):
    """Human-readable explanation generated from the model prediction and supplied text."""

    summary: str = Field(description="A concise Chinese summary of the overall fit.")
    strengths: list[str] = Field(min_length=1, max_length=3)
    gaps: list[str] = Field(min_length=1, max_length=3)
    recommendations: list[str] = Field(min_length=1, max_length=3)
    disclaimer: str = Field(description="Human-review and non-decision disclaimer.")


class AnalysisResponse(BaseModel):
    """Deterministic match result plus an LLM-generated explanation."""

    match: MatchResponse
    report: AnalysisReport
