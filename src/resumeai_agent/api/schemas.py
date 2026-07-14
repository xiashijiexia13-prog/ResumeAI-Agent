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
