"""Download, validate, and clean the Resume-ATS Score dataset."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd


DATASET_NAME = "0xnbk/resume-ats-score-v1-en"
REQUIRED_COLUMNS = {"text", "ats_score", "original_label"}
LABELS = {"No Fit", "Potential Fit", "Good Fit"}
PAIR_SEPARATOR = re.compile(r"\s+\[?SEP\]?\s+", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Apply only safe whitespace normalization; preserve technical punctuation."""
    return " ".join(text.replace("\x00", " ").split())


def clean_split(frame: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate one source split and return clean resume/job pairs plus statistics."""
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{split_name} 缺少必要字段: {sorted(missing_columns)}")

    stats = {"source_rows": len(frame)}
    work = frame.loc[:, ["text", "ats_score", "original_label"]].dropna().copy()
    stats["rows_after_missing_value_removal"] = len(work)

    parts = work["text"].astype(str).str.split(PAIR_SEPARATOR, n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(f"{split_name} 中存在无法按 [SEP] 拆分的文本")

    work["resume_text"] = parts[0].map(clean_text)
    work["job_description_text"] = parts[1].map(clean_text)
    work["label"] = work["original_label"].astype(str).str.strip()
    work["ats_score"] = pd.to_numeric(work["ats_score"], errors="coerce")
    work = work.dropna(subset=["ats_score"])
    work = work[
        work["label"].isin(LABELS)
        & work["ats_score"].between(0, 100)
        & work["resume_text"].ne("")
        & work["job_description_text"].ne("")
    ].copy()
    stats["rows_after_validation"] = len(work)

    duplicate_count = int(work.duplicated(["resume_text", "job_description_text"]).sum())
    stats["duplicate_pairs_removed"] = duplicate_count
    work = work.drop_duplicates(["resume_text", "job_description_text"], keep="first")
    stats["rows_after_within_split_deduplication"] = len(work)
    work.insert(0, "source_split", split_name)
    return work[["source_split", "resume_text", "job_description_text", "label", "ats_score"]], stats


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    for directory in (raw_dir, processed_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_dir = raw_dir / ".hf_cache"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    source_files = [
        filename
        for filename in api.list_repo_files(DATASET_NAME, repo_type="dataset")
        if filename in {"train.csv", "validation.csv"}
    ]
    if set(source_files) != {"train.csv", "validation.csv"}:
        raise ValueError("数据集仓库中未找到预期的 train.csv 和 validation.csv")

    cleaned_splits: dict[str, pd.DataFrame] = {}
    report: dict[str, object] = {"dataset": DATASET_NAME, "splits": {}}

    for filename in source_files:
        split_name = Path(filename).stem
        downloaded_file = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(cache_dir / "hub"),
        )
        raw_path = raw_dir / f"resume_ats_{split_name}.csv"
        shutil.copy2(downloaded_file, raw_path)
        raw_frame = pd.read_csv(raw_path)
        cleaned_frame, stats = clean_split(raw_frame, split_name)
        cleaned_splits[split_name] = cleaned_frame
        report["splits"][split_name] = stats

    if {"train", "validation"}.issubset(cleaned_splits):
        train_pairs = set(zip(cleaned_splits["train"]["resume_text"], cleaned_splits["train"]["job_description_text"]))
        validation_pairs = set(zip(cleaned_splits["validation"]["resume_text"], cleaned_splits["validation"]["job_description_text"]))
        overlap_pairs = train_pairs & validation_pairs
        report["cross_split_overlap_detected"] = len(overlap_pairs)
        if overlap_pairs:
            keep_mask = ~cleaned_splits["train"].apply(
                lambda row: (row["resume_text"], row["job_description_text"]) in overlap_pairs,
                axis=1,
            )
            cleaned_splits["train"] = cleaned_splits["train"].loc[keep_mask].copy()
        report["cross_split_overlap_removed_from_train"] = len(overlap_pairs)

    for split_name, cleaned_frame in cleaned_splits.items():
        split_report = report["splits"][split_name]
        split_report["final_rows"] = len(cleaned_frame)
        split_report["label_distribution"] = cleaned_frame["label"].value_counts().to_dict()
        split_report["score_min"] = float(cleaned_frame["ats_score"].min())
        split_report["score_max"] = float(cleaned_frame["ats_score"].max())
        cleaned_frame.to_csv(processed_dir / f"resume_job_{split_name}_clean.csv", index=False)

    with (artifacts_dir / "data_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
