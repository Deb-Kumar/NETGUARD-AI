"""Model training pipelines for NETGUARD AI binary intrusion classification."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.loader import load_training_and_testing_sets
from src.data.validator import extract_features_and_target
from src.models import MODEL_BUILDERS, MODEL_SPECS, save_model_artifact
from src.preprocessing import (
    analyze_class_imbalance,
    build_full_preprocessing_pipeline,
    save_preprocessing_config,
)
from src.utils.config import (
    BINARY_TARGET_COLUMN,
    ProjectPaths,
    RANDOM_STATE,
    ensure_directories,
    get_paths,
)


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert estimator hyperparameters into JSON-serializable primitives."""
    safe: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


def train_single_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a single model pipeline strictly on training data and record metadata."""
    if model_key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model key '{model_key}'. Expected one of {list(MODEL_BUILDERS)}."
        )

    builder = MODEL_BUILDERS[model_key]
    spec = MODEL_SPECS[model_key]
    pipeline = builder(
        X_train,
        enable_feature_engineering=enable_feature_engineering,
        task=task,
    )
    if params:
        pipeline.set_params(**params)

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    duration_sec = time.perf_counter() - start

    clf_params = pipeline.named_steps["classifier"].get_params()
    record: dict[str, Any] = {
        "model_key": model_key,
        "model_name": spec.display_name,
        "model_family": spec.model_family,
        "task": task,
        "training_duration_seconds": round(duration_sec, 3),
        "training_samples": int(len(X_train)),
        "input_feature_count": int(X_train.shape[1]),
        "input_feature_names": list(X_train.columns),
        "parameters": _json_safe_params(clf_params),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return pipeline, record


def evaluate_class_weight_effect(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 3,
    max_samples: int = 40_000,
) -> dict[str, Any]:
    """Compare class_weight=None vs 'balanced' using Stratified CV on training data."""
    if len(X_train) > max_samples:
        # Stratified subsample on training data only for lightweight effect check
        combined = X_train.copy()
        combined["_target"] = y_train.to_numpy()
        frac = max_samples / len(combined)
        sampled = pd.concat(
            [
                grp.sample(n=max(1, int(round(len(grp) * frac))), random_state=RANDOM_STATE)
                for _, grp in combined.groupby("_target", sort=False)
            ],
            ignore_index=True,
        )
        X_sub = sampled.drop(columns=["_target"])
        y_sub = sampled["_target"]
        sampled_note = f"Stratified training subsample ({len(X_sub):,} rows)"
    else:
        X_sub, y_sub = X_train, y_train
        sampled_note = f"Full training set ({len(X_sub):,} rows)"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    comparison: dict[str, dict[str, dict[str, float]]] = {}

    for model_key in ("logistic_regression", "decision_tree"):
        comparison[model_key] = {}
        for weight_opt in (None, "balanced"):
            label = "none" if weight_opt is None else "balanced"
            pipe = MODEL_BUILDERS[model_key](
                X_sub,
                class_weight=weight_opt,
                enable_feature_engineering=True,
                task="binary",
            )
            scores = cross_validate(
                pipe,
                X_sub,
                y_sub,
                cv=skf,
                scoring=["recall", "precision", "f1", "accuracy"],
                n_jobs=1,
            )
            comparison[model_key][label] = {
                "recall_mean": round(float(scores["test_recall"].mean()), 4),
                "precision_mean": round(float(scores["test_precision"].mean()), 4),
                "f1_mean": round(float(scores["test_f1"].mean()), 4),
                "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
            }

    return {
        "cv_folds": n_splits,
        "dataset_scope": sampled_note,
        "results": comparison,
    }


def train_all_binary_models(
    train_df: pd.DataFrame | None = None,
    *,
    model_keys: Sequence[str] | None = None,
    enable_feature_engineering: bool = True,
    run_imbalance_study: bool = True,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Train all three binary intrusion detection models on training data only."""
    resolved_paths = ensure_directories(paths or get_paths())
    if train_df is None:
        train_df, _ = load_training_and_testing_sets(paths=resolved_paths)

    X_train, y_train = extract_features_and_target(train_df, task="binary")

    # Persist preprocessing configuration & class imbalance audit
    _, num_cols, cat_cols = build_full_preprocessing_pipeline(
        X_train,
        model_family="linear",
        enable_feature_engineering=enable_feature_engineering,
        task="binary",
    )
    imbalance_report = analyze_class_imbalance(y_train)
    prep_config_path = save_preprocessing_config(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
        imbalance_report=imbalance_report,
        enable_feature_engineering=enable_feature_engineering,
        paths=resolved_paths,
    )

    selected_keys = list(model_keys) if model_keys is not None else list(MODEL_BUILDERS.keys())
    trained_records: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, str] = {}

    for key in selected_keys:
        pipeline, record = train_single_model(
            key,
            X_train,
            y_train,
            enable_feature_engineering=enable_feature_engineering,
            task="binary",
        )
        out_path = resolved_paths.binary_models_dir / f"{key}.joblib"
        save_model_artifact(pipeline, out_path)
        record["artifact_path"] = str(out_path)
        trained_records[key] = record
        artifact_paths[key] = str(out_path)

    class_weight_effect = (
        evaluate_class_weight_effect(X_train, y_train)
        if run_imbalance_study
        else None
    )

    summary = {
        "task": "binary",
        "target_column": BINARY_TARGET_COLUMN,
        "dataset_version": "UNSW-NB15 predefined training set",
        "training_samples": int(len(X_train)),
        "input_features": list(X_train.columns),
        "preprocessing_config_path": str(prep_config_path),
        "class_imbalance": imbalance_report,
        "class_weight_study": class_weight_effect,
        "models": trained_records,
        "artifact_paths": artifact_paths,
    }

    summary_path = resolved_paths.model_metadata_dir / "binary_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
