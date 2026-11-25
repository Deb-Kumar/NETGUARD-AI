"""Inference engine supporting single-record and chunked batch CSV predictions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.loader import get_feature_metadata_map
from src.data.validator import (
    PREDEFINED_CATEGORICAL_COLUMNS,
    PREDEFINED_FEATURE_COLUMNS,
)
from src.evaluation.evaluate import evaluate_binary_model
from src.models import MODEL_SPECS, load_model_artifact
from src.utils.config import (
    BINARY_CLASS_NAMES,
    BINARY_TARGET_COLUMN,
    ID_COLUMN,
    MAX_UPLOAD_SIZE_BYTES,
    MULTICLASS_TARGET_COLUMN,
    ProjectPaths,
    get_paths,
)


class PredictionValidationError(ValueError):
    """Raised when uploaded CSV files or single-record inputs fail validation."""


@dataclass
class SinglePredictionResult:
    """Structured output for single-record traffic classification."""

    predicted_label: int
    predicted_class: str
    malicious_probability: float | None
    normal_probability: float | None
    model_name: str
    model_key: str
    timestamp_utc: str
    warnings: list[str] = field(default_factory=list)
    multiclass_attack_category: str | None = None
    multiclass_confidence: float | None = None


@dataclass
class BatchPredictionResult:
    """Structured output for batch CSV network flow prediction."""

    predictions_df: pd.DataFrame
    total_records: int
    normal_count: int
    malicious_count: int
    model_name: str
    model_key: str
    timestamp_utc: str
    inference_duration_ms: float
    has_ground_truth: bool
    ground_truth_metrics: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def get_expected_feature_schema(paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Return expected feature columns, types, and descriptions from metadata."""
    resolved_paths = paths or get_paths()
    meta_file = resolved_paths.model_metadata_dir / "model_metadata.json"
    feature_names = list(PREDEFINED_FEATURE_COLUMNS)
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        feature_names = list(meta.get("feature_names", feature_names))

    try:
        descriptions = get_feature_metadata_map(paths=resolved_paths)
    except Exception:
        descriptions = {}

    categorical = [c for c in feature_names if c in PREDEFINED_CATEGORICAL_COLUMNS]
    numerical = [c for c in feature_names if c not in categorical]
    return {
        "feature_names": feature_names,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "descriptions": descriptions,
    }


def load_active_binary_model(
    model_key: str = "best_model",
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Pipeline, str, str]:
    """Load the saved binary model pipeline without retraining."""
    resolved_paths = paths or get_paths()
    if model_key == "best_model":
        artifact = resolved_paths.binary_models_dir / "best_model.joblib"
        meta_path = resolved_paths.model_metadata_dir / "model_metadata.json"
        display_name = "Best Selected Model"
        resolved_key = "best_model"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            display_name = str(meta.get("selected_model_name", display_name))
            resolved_key = str(meta.get("selected_model_key", resolved_key))
    else:
        if model_key not in MODEL_SPECS:
            raise PredictionValidationError(
                f"Unknown binary model '{model_key}'. Choose from {list(MODEL_SPECS)}."
            )
        artifact = resolved_paths.binary_models_dir / f"{model_key}.joblib"
        display_name = MODEL_SPECS[model_key].display_name
        resolved_key = model_key

    pipeline = load_model_artifact(artifact, paths=resolved_paths)
    return pipeline, resolved_key, display_name


def load_active_multiclass_model(
    *,
    paths: ProjectPaths | None = None,
) -> tuple[Pipeline, str] | None:
    """Load optional multiclass attack_cat classifier if available."""
    resolved_paths = paths or get_paths()
    artifact = resolved_paths.multiclass_models_dir / "best_model.joblib"
    if not artifact.is_file():
        return None
    pipeline = load_model_artifact(artifact, paths=resolved_paths)
    meta_path = resolved_paths.model_metadata_dir / "multiclass_metadata.json"
    model_name = "Multiclass Attack Classifier"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = str(meta.get("selected_model_name", model_name))
    return pipeline, model_name


def validate_and_align_features(
    df: pd.DataFrame,
    *,
    expected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate uploaded/input rows, detect missing columns, coerce types, and order columns."""
    if df.empty:
        raise PredictionValidationError("Input dataset contains zero records.")

    expected = expected_features or list(PREDEFINED_FEATURE_COLUMNS)
    missing_cols = [col for col in expected if col not in df.columns]
    if missing_cols:
        raise PredictionValidationError(
            f"Missing {len(missing_cols)} required feature column(s): {missing_cols}"
        )

    warnings: list[str] = []
    extra_cols = [
        c
        for c in df.columns
        if c not in expected and c not in (ID_COLUMN, BINARY_TARGET_COLUMN, MULTICLASS_TARGET_COLUMN)
    ]
    if extra_cols:
        warnings.append(
            f"Ignored {len(extra_cols)} unexpected column(s) not used by the model: {extra_cols[:8]}"
        )

    aligned = df[expected].copy()

    # Coerce numerical columns safely and reject unparseable text, negatives, or +/-inf
    for col in expected:
        if col in PREDEFINED_CATEGORICAL_COLUMNS:
            aligned[col] = aligned[col].fillna("-").astype(str).str.strip()
            aligned[col] = aligned[col].replace({"": "-", "nan": "-", "None": "-"})
        else:
            original_na = aligned[col].isna()
            coerced = pd.to_numeric(aligned[col], errors="coerce")
            bad_non_numeric = int((coerced.isna() & ~original_na).sum())
            if bad_non_numeric > 0:
                raise PredictionValidationError(
                    f"Column '{col}' contains {bad_non_numeric} non-numeric invalid value(s)."
                )
            arr = coerced.to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(arr).any():
                raise PredictionValidationError(
                    f"Column '{col}' contains infinite (+/-inf) values, which are invalid."
                )
            if np.nanmin(arr) < 0.0 if not np.isnan(arr).all() else False:
                neg_count = int((arr[~np.isnan(arr)] < 0.0).sum())
                raise PredictionValidationError(
                    f"Column '{col}' contains {neg_count} negative value(s); "
                    "UNSW-NB15 flow features must be >= 0."
                )
            if original_na.any():
                warnings.append(
                    f"Column '{col}' has {int(original_na.sum())} missing value(s); median imputation applied."
                )
            aligned[col] = coerced

    return aligned, warnings


def predict_single_record(
    record: dict[str, Any],
    *,
    model_key: str = "best_model",
    include_multiclass: bool = True,
    paths: ProjectPaths | None = None,
) -> SinglePredictionResult:
    """Validate and classify a single network flow record."""
    if not isinstance(record, dict) or not record:
        raise PredictionValidationError("Single record input must be a non-empty dictionary.")

    resolved_paths = paths or get_paths()
    schema = get_expected_feature_schema(resolved_paths)
    pipeline, resolved_key, display_name = load_active_binary_model(
        model_key=model_key, paths=resolved_paths
    )

    input_df = pd.DataFrame([record])
    aligned_df, warnings = validate_and_align_features(
        input_df, expected_features=schema["feature_names"]
    )

    pred_label = int(pipeline.predict(aligned_df)[0])
    pred_class = BINARY_CLASS_NAMES.get(pred_label, f"Class {pred_label}")

    mal_prob: float | None = None
    norm_prob: float | None = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(aligned_df)[0]
        classes = list(getattr(pipeline, "classes_", [0, 1]))
        if 1 in classes:
            mal_prob = round(float(probs[classes.index(1)]), 4)
        if 0 in classes:
            norm_prob = round(float(probs[classes.index(0)]), 4)

    warnings.append(
        "Advisory: ML predictions indicate statistical similarity to UNSW-NB15 patterns "
        "and do not guarantee the presence or absence of an actual cyberattack."
    )

    mc_cat: str | None = None
    mc_conf: float | None = None
    if include_multiclass:
        mc_loaded = load_active_multiclass_model(paths=resolved_paths)
        if mc_loaded is not None:
            mc_pipe, _ = mc_loaded
            mc_cat = str(mc_pipe.predict(aligned_df)[0])
            if hasattr(mc_pipe, "predict_proba"):
                mc_probs = mc_pipe.predict_proba(aligned_df)[0]
                mc_conf = round(float(np.max(mc_probs)), 4)

    return SinglePredictionResult(
        predicted_label=pred_label,
        predicted_class=pred_class,
        malicious_probability=mal_prob,
        normal_probability=norm_prob,
        model_name=display_name,
        model_key=resolved_key,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
        multiclass_attack_category=mc_cat,
        multiclass_confidence=mc_conf,
    )


def predict_batch_csv(
    csv_source: Path | str | Any,
    *,
    filename: str | None = None,
    file_size_bytes: int | None = None,
    model_key: str = "best_model",
    chunksize: int = 25_000,
    paths: ProjectPaths | None = None,
) -> BatchPredictionResult:
    """Validate and classify a batch network flow CSV safely in chunks."""
    resolved_paths = paths or get_paths()

    if isinstance(csv_source, (str, Path)):
        path = Path(csv_source)
        if not path.is_file():
            raise PredictionValidationError(f"Uploaded CSV file not found: '{path.name}'.")
        filename = filename or path.name
        if file_size_bytes is None:
            file_size_bytes = path.stat().st_size

    if filename and not filename.lower().endswith(".csv"):
        raise PredictionValidationError(
            f"Unsupported file type for '{filename}'. Only .csv files are accepted."
        )
    if file_size_bytes is not None and file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise PredictionValidationError(
            f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    try:
        if hasattr(csv_source, "seek"):
            csv_source.seek(0)
        raw_df = pd.read_csv(csv_source, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(csv_source, "seek"):
            csv_source.seek(0)
        raw_df = pd.read_csv(csv_source, encoding="latin1", low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise PredictionValidationError("Uploaded CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise PredictionValidationError(f"Malformed CSV file: {exc}") from exc

    schema = get_expected_feature_schema(resolved_paths)
    aligned_df, warnings = validate_and_align_features(
        raw_df, expected_features=schema["feature_names"]
    )
    pipeline, resolved_key, display_name = load_active_binary_model(
        model_key=model_key, paths=resolved_paths
    )

    start_ns = time.perf_counter_ns()
    preds_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    has_proba = hasattr(pipeline, "predict_proba")
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1

    for start_idx in range(0, len(aligned_df), chunksize):
        chunk = aligned_df.iloc[start_idx : start_idx + chunksize]
        preds_list.append(np.asarray(pipeline.predict(chunk), dtype=int))
        if has_proba:
            probs = pipeline.predict_proba(chunk)
            probs_list.append(np.asarray(probs[:, pos_idx], dtype=np.float64))

    duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    all_preds = np.concatenate(preds_list) if preds_list else np.empty(0, dtype=int)
    all_probs = np.concatenate(probs_list) if probs_list else None

    out_df = raw_df.copy()
    out_df["predicted_label"] = all_preds
    out_df["predicted_class"] = [
        BINARY_CLASS_NAMES.get(int(val), f"Class {int(val)}") for val in all_preds
    ]
    if all_probs is not None:
        out_df["malicious_probability"] = np.round(all_probs, 4)

    normal_count = int((all_preds == 0).sum())
    malicious_count = int((all_preds == 1).sum())

    # Optional ground-truth evaluation if actual 'label' column is supplied
    has_ground_truth = False
    gt_metrics: dict[str, Any] | None = None
    if BINARY_TARGET_COLUMN in raw_df.columns:
        gt_series = pd.to_numeric(raw_df[BINARY_TARGET_COLUMN], errors="coerce")
        if not gt_series.isna().any() and set(gt_series.astype(int).unique()).issubset({0, 1}):
            has_ground_truth = True
            gt_metrics = evaluate_binary_model(
                pipeline,
                aligned_df,
                gt_series.astype(int),
                model_key=resolved_key,
                model_name=display_name,
                split_name="uploaded_ground_truth",
            )
        else:
            warnings.append(
                "Ground-truth column 'label' contained invalid or missing values; skipped metrics."
            )

    warnings.append(
        "Predicted classifications are automated estimates and must not be treated as verified attack labels."
    )

    return BatchPredictionResult(
        predictions_df=out_df,
        total_records=int(len(out_df)),
        normal_count=normal_count,
        malicious_count=malicious_count,
        model_name=display_name,
        model_key=resolved_key,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        inference_duration_ms=round(duration_ms, 2),
        has_ground_truth=has_ground_truth,
        ground_truth_metrics=gt_metrics,
        warnings=warnings,
    )
