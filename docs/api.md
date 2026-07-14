# FastAPI 模型接口

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn resumeai_agent.api.main:app --reload
```

启动后访问 `http://127.0.0.1:8000/docs`，可使用 FastAPI 自动生成的 Swagger 文档测试接口。

## 健康检查

`GET /health`

用于部署平台、负载均衡器或监控系统确认服务已经启动并能加载当前模型。

## 岗位匹配

`POST /api/v1/match`

请求体：

```json
{
  "resume_text": "Python developer with FastAPI experience",
  "job_description_text": "Backend developer requiring Python and API skills"
}
```

响应体：

```json
{
  "label": "Good Fit",
  "probabilities": {
    "No Fit": 0.12,
    "Potential Fit": 0.23,
    "Good Fit": 0.65
  },
  "model_name": "TF-IDF + Logistic Regression"
}
```

概率是当前模型的分类置信度，不是录用概率，也不能代替人工招聘决定。
