# 数据源说明

## 主数据集

- 名称：Resume-ATS Score Dataset v1 (English)
- 地址：https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en
- 许可证：Apache-2.0
- 数据规模：6,374 条英文简历—岗位描述配对样本（训练集 5,099 条，验证集 1,275 条）

## 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `text` | string | 以 `[SEP]` 分隔的“简历文本 + 岗位描述文本” |
| `ats_score` | float | 18.3–90.7 的连续匹配分数 |
| `original_label` | string | `No Fit`、`Potential Fit` 或 `Good Fit` |

## 选择原因

同一份数据同时提供三档匹配标签和连续分数。因此项目可以先训练 TF-IDF + Logistic Regression 三分类基线模型，再用连续分数训练/评估 Sentence Transformer 的语义相似度模型，并在相同任务上比较效果。

## 使用边界

数据集为英文，且匹配分数来自算法计算而非人工招聘决策。它仅用于学习、研究和作品集演示，系统输出应辅助人工判断，不能用于自动化招聘淘汰或录用决定。
