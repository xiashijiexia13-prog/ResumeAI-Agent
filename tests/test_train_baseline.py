import pandas as pd

from resumeai_agent.models.train_baseline import build_pipeline, combine_pair_text


def test_baseline_pipeline_fits_and_predicts_known_labels() -> None:
    examples = pd.DataFrame(
        {
            "resume_text": [
                "Python FastAPI developer",
                "Python API engineer",
                "React frontend developer",
                "JavaScript UI engineer",
                "Registered nurse ICU care",
                "Clinical nurse patient care",
            ],
            "job_description_text": [
                "Python FastAPI backend role",
                "API engineer Python role",
                "React web interface role",
                "JavaScript frontend role",
                "ICU nursing position",
                "patient care nurse role",
            ],
        }
    )
    labels = pd.Series(
        ["Good Fit", "Good Fit", "Potential Fit", "Potential Fit", "No Fit", "No Fit"]
    )

    model = build_pipeline()
    model.fit(combine_pair_text(examples), labels)

    predictions = model.predict(combine_pair_text(examples))
    assert set(predictions).issubset({"No Fit", "Potential Fit", "Good Fit"})