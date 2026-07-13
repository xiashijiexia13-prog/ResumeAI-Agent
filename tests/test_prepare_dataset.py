import pandas as pd

from resumeai_agent.data.prepare_dataset import clean_split


def test_clean_split_removes_duplicate_pairs_and_invalid_labels() -> None:
    source = pd.DataFrame(
        {
            "text": [
                "Python developer [SEP] Python backend engineer",
                "Python developer [SEP] Python backend engineer",
                "Nurse [SEP] Backend engineer",
            ],
            "ats_score": [82.0, 82.0, 20.0],
            "original_label": ["Good Fit", "Good Fit", "Unknown"],
        }
    )

    cleaned, stats = clean_split(source, "train")

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["resume_text"] == "Python developer"
    assert cleaned.iloc[0]["job_description_text"] == "Python backend engineer"
    assert stats["duplicate_pairs_removed"] == 1
