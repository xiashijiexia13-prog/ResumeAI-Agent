# LLM 匹配分析报告

`POST /api/v1/analyze` 先运行本地 TF-IDF 匹配模型，再将模型标签、概率、简历和 JD 交给 OpenAI Responses API，生成中文结构化报告。

## 本地配置

1. 复制 `.env.example` 为 `.env`。
2. 在 `.env` 中填写 `OPENAI_API_KEY`。
3. 可选：修改 `OPENAI_MODEL`。默认值为 `gpt-5.6-luna`。
4. 重启 Uvicorn 服务。

`.env` 已被 Git 忽略，绝不能提交 API Key。

## 请求与响应

请求仍使用与 `/api/v1/match` 相同的输入：

```json
{
  "resume_text": "Python developer with FastAPI experience",
  "job_description_text": "Backend developer requiring Python and API skills"
}
```

响应包含确定性的 `match` 模型结果，以及由 LLM 生成的 `report`：

```json
{
  "match": {
    "label": "Good Fit",
    "probabilities": {"No Fit": 0.1, "Potential Fit": 0.2, "Good Fit": 0.7},
    "model_name": "TF-IDF + Logistic Regression"
  },
  "report": {
    "summary": "候选人与岗位存在较高的技能相关性。",
    "strengths": ["Python 与 API 开发经验"],
    "gaps": ["未明确体现云部署经验"],
    "recommendations": ["补充可量化的 API 项目成果"],
    "disclaimer": "本报告仅供人工辅助参考，不能用于自动招聘决策。"
  }
}
```

## 安全与边界

- API Key 只存放在本地 `.env` 或部署平台的密钥管理系统中。
- 服务不会记录简历和 JD 正文。
- 每段输入限制为最多 8,000 个字符再发送给 LLM，以限制成本和上下文长度。
- LLM 只能解释匹配结果，不能做自动录用、淘汰或其他招聘决定。
