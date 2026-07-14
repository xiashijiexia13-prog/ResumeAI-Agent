import pytest

from resumeai_agent.api.schemas import AnalysisReport
from resumeai_agent.services.reporting import OpenAIReportGenerator, ReportGenerationError, truncate_text


class StubResponses:
    def parse(self, **kwargs: object) -> object:
        class Response:
            output_parsed = AnalysisReport(
                summary="匹配分析摘要。",
                strengths=["Python"],
                gaps=["Docker"],
                recommendations=["补充 Docker 项目"],
                disclaimer="仅供人工辅助参考。",
            )

        return Response()


class StubOpenAIClient:
    responses = StubResponses()


def test_report_generator_returns_validated_structured_report() -> None:
    generator = OpenAIReportGenerator(client=StubOpenAIClient(), model="test-model")

    report = generator.generate(
        "Python developer",
        "Backend role requiring Python",
        {"label": "Good Fit", "probabilities": {"Good Fit": 0.7}},
    )

    assert report.summary == "匹配分析摘要。"


def test_report_generator_requires_api_key_when_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = OpenAIReportGenerator()

    with pytest.raises(ReportGenerationError, match="OPENAI_API_KEY"):
        generator.generate("resume", "job", {"label": "No Fit", "probabilities": {}})


def test_truncate_text_bounds_prompt_length() -> None:
    truncated = truncate_text("a" * 8_001)

    assert truncated.startswith("a" * 8_000)
    assert "被截断" in truncated
