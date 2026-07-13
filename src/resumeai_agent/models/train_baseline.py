"""Train and evaluate the TF-IDF + Logistic Regression baseline classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]
RANDOM_STATE = 42


def combine_pair_text(frame: pd.DataFrame) -> pd.Series:
    """Create one explicit text input from a resume and its job description."""
    return (
        "resume: "
        + frame["resume_text"].astype(str)
        + " job_description: "
        + frame["job_description_text"].astype(str)
    )


def build_pipeline() -> Pipeline:
    """Build a reproducible, class-balanced text classification baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)[a-zA-Z0-9][a-zA-Z0-9+#.\-]*",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, features: pd.Series, labels: pd.Series) -> dict[str, object]:
    """Return classification metrics in a JSON-serializable structure."""
    predictions = model.predict(features)
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(labels, predictions, average="weighted")), 4),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": LABEL_ORDER,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABEL_ORDER).tolist(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    train_path = project_root / "data" / "processed" / "resume_job_train_clean.csv"
    test_path = project_root / "data" / "processed" / "resume_job_validation_clean.csv"
    model_dir = project_root / "artifacts" / "models"
    metrics_path = project_root / "artifacts" / "baseline_metrics.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    training_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        combine_pair_text(training_data),
        training_data["label"],
        test_size=0.2,
        stratify=training_data["label"],
        random_state=RANDOM_STATE,
    )

    validation_model = build_pipeline()
    validation_model.fit(train_features, train_labels)
    validation_metrics = evaluate(validation_model, validation_features, validation_labels)

    final_model = build_pipeline()
    final_model.fit(combine_pair_text(training_data), training_data["label"])
    test_metrics = evaluate(final_model, combine_pair_text(test_data), test_data["label"])

    joblib.dump(final_model, model_dir / "tfidf_logistic_regression.joblib")
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "random_state": RANDOM_STATE,
        "train_rows_before_validation_split": len(training_data),
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_data),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
