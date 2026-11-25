"""Reusable, leakage-free Scikit-learn preprocessing pipelines for NETGUARD AI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data.validator import PREDEFINED_FEATURE_COLUMNS, validate_no_target_leakage
from src.utils.config import (
    NON_FEATURE_COLUMNS,
    ProjectPaths,
    ensure_directories,
    get_paths,
)

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "total_bytes",
    "total_pkts",
    "byte_ratio",
    "pkt_ratio",
    "avg_bytes_per_pkt",
    "ttl_diff",
    "loss_total",
    "is_service_unspecified",
)


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise, stateless cybersecurity flow feature engineering transformer.

    All derived features depend strictly on values within a single network flow
    record, preventing any cross-row or train/test data leakage.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y: Any = None) -> NetworkFeatureEngineer:
        """Fit stateless transformer and record input schema."""
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = list(X.columns)
        else:
            self.input_columns_ = [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append domain-derived flow features to a copy of the input DataFrame."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.input_columns_)
        if not self.enabled:
            return df

        if {"sbytes", "dbytes"}.issubset(df.columns):
            sbytes = pd.to_numeric(df["sbytes"], errors="coerce").fillna(0.0).astype(np.float64)
            dbytes = pd.to_numeric(df["dbytes"], errors="coerce").fillna(0.0).astype(np.float64)
            df["total_bytes"] = sbytes + dbytes
            df["byte_ratio"] = sbytes / (np.maximum(dbytes, 0.0) + 1.0)
        else:
            sbytes = dbytes = None

        if {"spkts", "dpkts"}.issubset(df.columns):
            spkts = pd.to_numeric(df["spkts"], errors="coerce").fillna(0.0).astype(np.float64)
            dpkts = pd.to_numeric(df["dpkts"], errors="coerce").fillna(0.0).astype(np.float64)
            df["total_pkts"] = spkts + dpkts
            df["pkt_ratio"] = spkts / (np.maximum(dpkts, 0.0) + 1.0)
        else:
            spkts = dpkts = None

        if sbytes is not None and dbytes is not None and spkts is not None and dpkts is not None:
            df["avg_bytes_per_pkt"] = (sbytes + dbytes) / (np.maximum(spkts + dpkts, 0.0) + 1.0)

        if {"sttl", "dttl"}.issubset(df.columns):
            sttl = pd.to_numeric(df["sttl"], errors="coerce").fillna(0.0).astype(np.float64)
            dttl = pd.to_numeric(df["dttl"], errors="coerce").fillna(0.0).astype(np.float64)
            df["ttl_diff"] = sttl - dttl

        if {"sloss", "dloss"}.issubset(df.columns):
            sloss = pd.to_numeric(df["sloss"], errors="coerce").fillna(0.0).astype(np.float64)
            dloss = pd.to_numeric(df["dloss"], errors="coerce").fillna(0.0).astype(np.float64)
            df["loss_total"] = sloss + dloss

        if "service" in df.columns:
            df["is_service_unspecified"] = (
                df["service"].astype(str).str.strip() == "-"
            ).astype(np.int8)

        return df

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return output column names for Scikit-learn introspection."""
        base = list(input_features) if input_features is not None else list(self.input_columns_)
        if not self.enabled:
            return np.asarray(base, dtype=object)
        extra = [c for c in ENGINEERED_FEATURE_NAMES if c not in base]
        return np.asarray(base + extra, dtype=object)


def split_feature_columns(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    task: Literal["binary", "multiclass"] = "binary",
) -> tuple[list[str], list[str]]:
    """Automatically separate numerical and categorical columns with leakage checks."""
    candidates = (
        list(feature_columns)
        if feature_columns is not None
        else [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    )
    validate_no_target_leakage(candidates, task=task)

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scale_numeric: bool = True,
    sparse_output: bool = True,
) -> ColumnTransformer:
    """Create a ColumnTransformer for numerical and categorical preprocessing."""
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, list(numerical_columns)),
            ("categorical", cat_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=0.3 if sparse_output else 0.0,
    )


def build_full_preprocessing_pipeline(
    sample_df: pd.DataFrame,
    *,
    model_family: Literal["linear", "tree"] = "linear",
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
    sparse_output: bool = True,
) -> tuple[Pipeline, list[str], list[str]]:
    """Build an end-to-end leakage-free preprocessing Pipeline (engineering + ColumnTransformer).

    Args:
        sample_df: Training DataFrame (never test data) used only to inspect column dtypes.
        model_family: ``'linear'`` applies StandardScaler; ``'tree'`` skips scaling.
        enable_feature_engineering: Whether to append row-wise engineered flow features.
        task: Classification task (``'binary'`` or ``'multiclass'``).
        sparse_output: Whether OneHotEncoder emits sparse CSR matrices.
    """
    base_cols = [c for c in PREDEFINED_FEATURE_COLUMNS if c in sample_df.columns]
    if not base_cols:
        base_cols = [c for c in sample_df.columns if c not in NON_FEATURE_COLUMNS]
    validate_no_target_leakage(base_cols, task=task)

    engineer = NetworkFeatureEngineer(enabled=enable_feature_engineering)
    preview_df = engineer.fit_transform(sample_df[base_cols].head(5))
    num_cols, cat_cols = split_feature_columns(preview_df, task=task)

    preprocessor = build_preprocessor(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
        scale_numeric=(model_family == "linear"),
        sparse_output=sparse_output,
    )
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", engineer),
            ("column_transformer", preprocessor),
        ]
    )
    return pipeline, num_cols, cat_cols


def analyze_class_imbalance(y: pd.Series) -> dict[str, Any]:
    """Quantify target class balance and compute balanced weights without resampling."""
    counts = y.value_counts().sort_index()
    total = int(len(y))
    classes = counts.index.tolist()
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(classes),
        y=y.to_numpy(),
    )
    max_cnt = int(counts.max()) if not counts.empty else 0
    min_cnt = int(counts.min()) if not counts.empty else 1

    return {
        "total_samples": total,
        "class_counts": {str(cls): int(cnt) for cls, cnt in counts.items()},
        "class_percentages": {
            str(cls): round(float(cnt) / total * 100.0, 2) if total else 0.0
            for cls, cnt in counts.items()
        },
        "imbalance_ratio_max_to_min": round(float(max_cnt) / max(min_cnt, 1), 3),
        "balanced_class_weights": {
            str(cls): round(float(w), 4) for cls, w in zip(classes, weights)
        },
        "smote_policy": (
            "SMOTE is intentionally not auto-applied before splitting or on the full 175k "
            "one-hot encoded matrix due to synthetic flow artifact risks and memory cost; "
            "class_weight='balanced' is evaluated natively inside models."
        ),
    }


def save_preprocessing_config(
    *,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    imbalance_report: dict[str, Any],
    enable_feature_engineering: bool = True,
    paths: ProjectPaths | None = None,
    filename: str = "preprocessing_config.json",
) -> Path:
    """Persist preprocessing configuration metadata cleanly to models/metadata/."""
    resolved_paths = ensure_directories(paths or get_paths())
    config_payload = {
        "excluded_columns": list(NON_FEATURE_COLUMNS),
        "base_feature_columns": list(PREDEFINED_FEATURE_COLUMNS),
        "feature_engineering_enabled": enable_feature_engineering,
        "engineered_features": list(ENGINEERED_FEATURE_NAMES) if enable_feature_engineering else [],
        "numerical_columns": list(numerical_columns),
        "categorical_columns": list(categorical_columns),
        "pipelines": {
            "linear": {
                "numerical": ["SimpleImputer(strategy='median')", "StandardScaler()"],
                "categorical": [
                    "SimpleImputer(strategy='most_frequent')",
                    "OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
                ],
            },
            "tree": {
                "numerical": ["SimpleImputer(strategy='median')"],
                "categorical": [
                    "SimpleImputer(strategy='most_frequent')",
                    "OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
                ],
            },
        },
        "class_imbalance_analysis": imbalance_report,
    }
    out_path = resolved_paths.model_metadata_dir / filename
    out_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return out_path
