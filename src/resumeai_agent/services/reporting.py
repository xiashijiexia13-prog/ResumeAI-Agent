"""Generate a bounded, structured LLM explanation for one match prediction."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from resumeai_agent.api.schemas import AnalysisReport


DEFAULT_MODEL = "gpt-5.6-luna"
MAX_TEXT_CHARS = 8_000

# Load local development secrets once. Production should provide environment variables directly.
load_dotenv()


class ReportGenerationError(RuntimeError):
    """Raised when a report cannot be generated safely or correctly."""


def truncate_text(text: str) -> str:
    """Bound prompt size without logging candidate text or sending unlimited input."""
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS]}\n[文本因长度限制被截断]"


class OpenAIReportGenerator:
    """OpenAI-backed generator that returns only the validated report schema."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    def generate(
        self,
        resume_text: str,
        job_description_text: str,
        match: dict[str, object],
    ) -> AnalysisReport:
        """Create Chinese guidance grounded only in supplied resume, JD, and model result."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ReportGenerationError(
                    "未配置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写 API Key。"
                )
            self._client = OpenAI(api_key=api_key)

        response = self._client.responses.parse(
            model=self._model,
            instructions=(
                "你是简历岗位匹配解释助手。仅依据提供的简历、岗位描述和模型结果输出中文报告。"
                "不得编造候选人经历、技能、学历或岗位要求；信息不足时明确说明。"
                "不要做录用、淘汰或法律意义的招聘决定。"
            ),
            input=(
                "模型匹配结果：\n"
                f"标签：{match['label']}\n"
                f"类别概率：{match['probabilities']}\n\n"
                f"简历：\n{truncate_text(resume_text)}\n\n"
                f"岗位描述：\n{truncate_text(job_description_text)}"
            ),
            text_format=AnalysisReport,
        )
        if response.output_parsed is None:
            raise ReportGenerationError("LLM 未返回符合预期结构的分析报告。")
        return response.output_parsed
