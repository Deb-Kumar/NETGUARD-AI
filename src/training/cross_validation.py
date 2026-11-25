"""Stratified K-Fold Cross-Validation strictly on training data."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.loader import load_training_and_testing_sets
from src.data.validator import extract_features_and_target
from src.models import MODEL_BUILDERS, MODEL_SPECS
from src.utils.config import (
    CV_FOLDS,
    ProjectPaths,
    RANDOM_STATE,
    ensure_directories,
    get_paths,
)


def _stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_samples: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Take a reproducible stratified subsample strictly from training data."""
    if len(X) <= max_samples:
        return X, y
    combined = X.copy()
    combined["_target"] = y.to_numpy()
    frac = max_samples / len(combined)
    sampled = pd.concat(
        [
            grp.sample(
                n=min(len(grp), max(1, int(round(len(grp) * frac)))),
                random_state=random_state,
            )
            for _, grp in combined.groupby("_target", sort=False)
        ],
        ignore_index=True,
    )
    return sampled.drop(columns=["_target"]), sampled["_target"]


def run_stratified_cv(
    train_df: pd.DataFrame | None = None,
    *,
    model_keys: Sequence[str] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    n_splits: int = CV_FOLDS,
    n_jobs: int = 1,
    max_samples: int | None = None,
    enable_feature_engineering: bool = True,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Run StratifiedKFold cross-validation across models on training data only."""
    resolved_paths = ensure_directories(paths or get_paths())
    if train_df is None:
        train_df, _ = load_training_and_testing_sets(paths=resolved_paths)

    X_train, y_train = extract_features_and_target(train_df, task="binary")

    if max_samples is not None and len(X_train) > max_samples:
        X_cv, y_cv = _stratified_subsample(X_train, y_train, max_samples=max_samples)
        sampling_note = (
            f"Stratified training subsample of {len(X_cv):,} rows out of {len(X_train):,} "
            f"(random_state={RANDOM_STATE})."
        )
    else:
        X_cv, y_cv = X_train, y_train
        sampling_note = f"Full training dataset ({len(X_cv):,} rows)."

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    selected_keys = list(model_keys) if model_keys is not None else list(MODEL_BUILDERS.keys())
    cv_models: dict[str, dict[str, Any]] = {}

    for key in selected_keys:
        if key not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model key '{key}'.")
        spec = MODEL_SPECS[key]
        pipeline = MODEL_BUILDERS[key](
            X_cv,
            enable_feature_engineering=enable_feature_engineering,
            task="binary",
        )
        if model_params and key in model_params and model_params[key]:
            pipeline.set_params(**model_params[key])

        start = time.perf_counter()
        raw_scores = cross_validate(
            pipeline,
            X_cv,
            y_cv,
            cv=skf,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        duration_sec = time.perf_counter() - start

        metrics_summary: dict[str, dict[str, Any]] = {}
        for metric_name in scoring:
            fold_arr = np.asarray(raw_scores[f"test_{metric_name}"], dtype=float)
            metrics_summary[metric_name] = {
                "fold_scores": [round(float(v), 4) for v in fold_arr],
                "mean": round(float(fold_arr.mean()), 4),
                "std": round(float(fold_arr.std(ddof=1)) if len(fold_arr) > 1 else 0.0, 4),
            }

        cv_models[key] = {
            "model_key": key,
            "model_name": spec.display_name,
            "duration_seconds": round(duration_sec, 3),
            "metrics": metrics_summary,
        }

    comparison_rows: list[dict[str, Any]] = []
    for key, entry in cv_models.items():
        m = entry["metrics"]
        comparison_rows.append(
            {
                "Model Key": key,
                "Model Name": entry["model_name"],
                "CV Recall Mean": m["recall"]["mean"],
                "CV Recall Std": m["recall"]["std"],
                "CV Precision Mean": m["precision"]["mean"],
                "CV F1 Mean": m["f1"]["mean"],
                "CV F1 Std": m["f1"]["std"],
                "CV Accuracy Mean": m["accuracy"]["mean"],
                "CV ROC-AUC Mean": m["roc_auc"]["mean"],
                "Duration (s)": entry["duration_seconds"],
            }
        )

    payload: dict[str, Any] = {
        "n_splits": n_splits,
        "random_state": RANDOM_STATE,
        "dataset_scope": sampling_note,
        "models": cv_models,
        "comparison_table": comparison_rows,
    }

    out_path = resolved_paths.model_metadata_dir / "cv_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["results_path"] = str(out_path)
    return payload
