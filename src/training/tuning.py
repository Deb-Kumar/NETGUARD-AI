"""Configurable GridSearchCV hyperparameter tuning strictly on training data."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.loader import load_training_and_testing_sets
from src.data.validator import extract_features_and_target
from src.models import (
    MODEL_BUILDERS,
    MODEL_SPECS,
    PARAM_GRID_BUILDERS,
    save_model_artifact,
)
from src.training.cross_validation import _stratified_subsample
from src.utils.config import (
    CV_FOLDS,
    ProjectPaths,
    RANDOM_STATE,
    ensure_directories,
    get_paths,
)


def _format_top_candidates(cv_results: dict[str, Any], primary_metric: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Extract top-k GridSearchCV parameter rows cleanly for display & audit."""
    df = pd.DataFrame(cv_results)
    rank_col = f"rank_test_{primary_metric}" if f"rank_test_{primary_metric}" in df.columns else "rank_test_score"
    if rank_col in df.columns:
        df = df.sort_values(rank_col)

    rows: list[dict[str, Any]] = []
    for _, row in df.head(top_k).iterrows():
        params = {
            str(k).replace("classifier__", ""): (
                None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            )
            for k, v in row["params"].items()
        }
        entry: dict[str, Any] = {
            "rank": int(row[rank_col]) if rank_col in row else len(rows) + 1,
            "params": params,
            "mean_fit_time_s": round(float(row["mean_fit_time"]), 3),
        }
        for metric_key in ("recall", "precision", "f1", "roc_auc", "accuracy"):
            col = f"mean_test_{metric_key}"
            if col in row:
                entry[f"cv_{metric_key}"] = round(float(row[col]), 4)
        rows.append(entry)
    return rows


def run_grid_search_tuning(
    train_df: pd.DataFrame | None = None,
    *,
    model_keys: Sequence[str] | None = None,
    cv_folds: int = CV_FOLDS,
    scoring: str = "f1",
    n_jobs: int = 1,
    custom_grids: dict[str, dict[str, list[Any]]] | None = None,
    max_tuning_samples: int | None = None,
    refit_full_train: bool = True,
    enable_feature_engineering: bool = True,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Run GridSearchCV on training data only and persist tuned binary models."""
    resolved_paths = ensure_directories(paths or get_paths())
    if train_df is None:
        train_df, _ = load_training_and_testing_sets(paths=resolved_paths)

    X_train, y_train = extract_features_and_target(train_df, task="binary")

    if max_tuning_samples is not None and len(X_train) > max_tuning_samples:
        X_search, y_search = _stratified_subsample(
            X_train, y_train, max_samples=max_tuning_samples
        )
        search_scope = (
            f"Stratified training subset of {len(X_search):,} rows out of {len(X_train):,} "
            f"(random_state={RANDOM_STATE}); final winner refitted on full {len(X_train):,} rows."
        )
    else:
        X_search, y_search = X_train, y_train
        search_scope = f"Full training dataset ({len(X_search):,} rows)."

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    multi_scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }
    if scoring not in multi_scoring:
        raise ValueError(
            f"Unsupported primary scoring '{scoring}'. Expected one of {list(multi_scoring)}."
        )

    selected_keys = list(model_keys) if model_keys is not None else list(MODEL_BUILDERS.keys())
    tuning_results: dict[str, dict[str, Any]] = {}

    for key in selected_keys:
        if key not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model key '{key}'.")
        spec = MODEL_SPECS[key]
        base_pipeline = MODEL_BUILDERS[key](
            X_search,
            enable_feature_engineering=enable_feature_engineering,
            task="binary",
        )
        grid = (
            custom_grids[key]
            if custom_grids and key in custom_grids
            else PARAM_GRID_BUILDERS[key]()
        )

        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=grid,
            cv=skf,
            scoring=multi_scoring,
            refit=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        start = time.perf_counter()
        search.fit(X_search, y_search)
        tuning_duration = time.perf_counter() - start

        best_idx = int(search.best_index_)
        cv_df = pd.DataFrame(search.cv_results_)
        best_cv_metrics = {
            metric_name: round(float(cv_df.loc[best_idx, f"mean_test_{metric_name}"]), 4)
            for metric_name in multi_scoring
        }

        # Refit best hyperparameters on 100% of X_train when search used a subsample
        final_pipeline = search.best_estimator_
        refit_duration = 0.0
        if refit_full_train and len(X_search) < len(X_train):
            refit_start = time.perf_counter()
            final_pipeline = MODEL_BUILDERS[key](
                X_train,
                enable_feature_engineering=enable_feature_engineering,
                task="binary",
            )
            final_pipeline.set_params(**search.best_params_)
            final_pipeline.fit(X_train, y_train)
            refit_duration = time.perf_counter() - refit_start

        artifact_path = resolved_paths.binary_models_dir / f"{key}.joblib"
        save_model_artifact(final_pipeline, artifact_path)

        clean_best_params = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        }

        tuning_results[key] = {
            "model_key": key,
            "model_name": spec.display_name,
            "best_params": clean_best_params,
            "primary_scoring": scoring,
            "best_cv_score": round(float(search.best_score_), 4),
            "best_cv_metrics": best_cv_metrics,
            "num_candidates": int(len(cv_df)),
            "tuning_duration_seconds": round(tuning_duration, 3),
            "full_refit_duration_seconds": round(refit_duration, 3),
            "top_candidates": _format_top_candidates(search.cv_results_, scoring, top_k=5),
            "artifact_path": str(artifact_path),
        }

    payload: dict[str, Any] = {
        "cv_folds": cv_folds,
        "primary_scoring": scoring,
        "n_jobs": n_jobs,
        "search_scope": search_scope,
        "final_refit_samples": int(len(X_train)) if refit_full_train else int(len(X_search)),
        "models": tuning_results,
    }

    out_path = resolved_paths.model_metadata_dir / "tuning_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["results_path"] = str(out_path)
    return payload
