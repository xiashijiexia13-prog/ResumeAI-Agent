from fastapi.testclient import TestClient

from resumeai_agent.api.main import create_app


class StubMatchingService:
    """Predictable replacement for the saved model in API contract tests."""

    def predict(self, resume_text: str, job_description_text: str) -> dict[str, object]:
        return {
            "label": "Good Fit",
            "probabilities": {"No Fit": 0.1, "Potential Fit": 0.2, "Good Fit": 0.7},
            "model_name": "TF-IDF + Logistic Regression",
        }


def test_health_returns_ready_status() -> None:
    with TestClient(create_app(StubMatchingService())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_match_returns_structured_prediction() -> None:
    with TestClient(create_app(StubMatchingService())) as client:
        response = client.post(
            "/api/v1/match",
            json={
                "resume_text": "Python developer with FastAPI experience",
                "job_description_text": "Backend developer requiring Python and API skills",
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "Good Fit"
    assert body["probabilities"]["Good Fit"] == 0.7


def test_match_rejects_empty_text() -> None:
    with TestClient(create_app(StubMatchingService())) as client:
        response = client.post(
            "/api/v1/match",
            json={"resume_text": "", "job_description_text": "Backend developer"},
        )

    assert response.status_code == 422
