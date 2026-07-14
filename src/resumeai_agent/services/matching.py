"""Load the deployed baseline model and expose one prediction operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from resumeai_agent.models.train_baseline import combine_pair_text


MODEL_NAME = "TF-IDF + Logistic Regression"


class MatchingService:
    """Adapter between raw API text and the persisted scikit-learn pipeline."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_artifact(cls, model_path: Path) -> "MatchingService":
        """Load the model once when the API process starts."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到模型文件：{model_path}。请先运行基线模型训练脚本。"
            )
        return cls(joblib.load(model_path))

    def predict(self, resume_text: str, job_description_text: str) -> dict[str, object]:
        """Return the best class and all class probabilities for one text pair."""
        frame = pd.DataFrame(
            [{"resume_text": resume_text, "job_description_text": job_description_text}]
        )
        features = combine_pair_text(frame)
        probabilities = self._model.predict_proba(features)[0]
        class_probabilities = {
            label: round(float(probability), 4)
            for label, probability in zip(self._model.classes_, probabilities, strict=True)
        }
        predicted_label = max(class_probabilities, key=class_probabilities.get)
        return {
            "label": predicted_label,
            "probabilities": class_probabilities,
            "model_name": MODEL_NAME,
        }
