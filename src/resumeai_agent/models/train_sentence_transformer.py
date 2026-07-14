"""Train a semantic resume-job classifier using Sentence Transformer embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from resumeai_agent.models.train_baseline import RANDOM_STATE, evaluate


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_pair_features(resume_embeddings: np.ndarray, job_embeddings: np.ndarray) -> np.ndarray:
    """Represent similarity with vector difference, interaction, and cosine similarity."""
    absolute_difference = np.abs(resume_embeddings - job_embeddings)
    elementwise_product = resume_embeddings * job_embeddings
    cosine_similarity = np.sum(resume_embeddings * job_embeddings, axis=1, keepdims=True)
    return np.hstack([absolute_difference, elementwise_product, cosine_similarity])


def build_classifier() -> OneVsRestClassifier:
    """Build a CPU-efficient multiclass classifier over semantic pair features."""
    return OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced", max_iter=300, random_state=RANDOM_STATE, solver="liblinear"
        )
    )


def encode_texts_with_cache(
    encoder: SentenceTransformer,
    texts: list[str],
    cache_dir: Path,
    cache_key: str,
    batch_size: int = 16,
    chunk_size: int = 250,
) -> np.ndarray:
    """Encode text in saved chunks so a long CPU job can resume after interruption."""
    cached_chunks: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        chunk_path = cache_dir / f"{cache_key}_{start}_{end}.npy"
        if chunk_path.exists():
            cached_chunks.append(np.load(chunk_path))
            continue

        print(f"正在编码 {cache_key}：第 {start + 1}–{end} 条...", flush=True)
        embeddings = encoder.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        np.save(chunk_path, embeddings)
        cached_chunks.append(embeddings)
    return np.vstack(cached_chunks)


def encode_pairs(
    encoder: SentenceTransformer, frame: pd.DataFrame, cache_dir: Path, split_name: str
) -> np.ndarray:
    """Encode resume and job texts separately, then create one feature vector per pair."""
    resume_embeddings = encode_texts_with_cache(
        encoder, frame["resume_text"].tolist(), cache_dir, f"{split_name}_resume"
    )
    job_embeddings = encode_texts_with_cache(
        encoder, frame["job_description_text"].tolist(), cache_dir, f"{split_name}_job"
    )
    return build_pair_features(resume_embeddings, job_embeddings)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    train_path = project_root / "data" / "processed" / "resume_job_train_clean.csv"
    test_path = project_root / "data" / "processed" / "resume_job_validation_clean.csv"
    artifact_dir = project_root / "artifacts"
    model_dir = artifact_dir / "models"
    cache_dir = artifact_dir / "model_cache"
    embedding_cache_dir = artifact_dir / "embedding_cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")

    training_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print("已读取清洗数据，开始划分内部训练集与验证集...", flush=True)
    train_data, validation_data = train_test_split(
        training_data,
        test_size=0.2,
        stratify=training_data["label"],
        random_state=RANDOM_STATE,
    )

    print(f"正在加载语义编码器：{MODEL_NAME}", flush=True)
    encoder = SentenceTransformer(MODEL_NAME, cache_folder=str(cache_dir))
    print("正在编码内部训练集...", flush=True)
    train_features = encode_pairs(encoder, train_data, embedding_cache_dir, "train")
    print("正在编码内部验证集...", flush=True)
    validation_features = encode_pairs(encoder, validation_data, embedding_cache_dir, "validation")
    print("正在编码最终测试集...", flush=True)
    test_features = encode_pairs(encoder, test_data, embedding_cache_dir, "test")

    validation_classifier = build_classifier()
    validation_classifier.fit(train_features, train_data["label"])
    validation_metrics = evaluate(
        validation_classifier, validation_features, validation_data["label"]
    )

    print("正在使用全部训练数据拟合最终分类器...", flush=True)
    final_features = np.vstack([train_features, validation_features])
    final_labels = pd.concat([train_data["label"], validation_data["label"]], ignore_index=True)
    final_classifier = build_classifier()
    final_classifier.fit(final_features, final_labels)
    test_metrics = evaluate(final_classifier, test_features, test_data["label"])

    encoder.save(str(model_dir / "sentence_transformer"))
    joblib.dump(final_classifier, model_dir / "semantic_logistic_regression.joblib")
    metrics = {
        "model": "all-MiniLM-L6-v2 embeddings + Logistic Regression",
        "encoder": MODEL_NAME,
        "embedding_dimension": 384,
        "pair_feature_dimension": int(final_features.shape[1]),
        "random_state": RANDOM_STATE,
        "train_rows_before_validation_split": len(training_data),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    with (artifact_dir / "sentence_transformer_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print("训练完成：模型与指标已保存到 artifacts/。", flush=True)


if __name__ == "__main__":
    main()
