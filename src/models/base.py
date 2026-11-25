"""Shared model pipeline assembly and safe Joblib artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_full_preprocessing_pipeline
from src.utils.config import ProjectPaths, ensure_directories, get_paths


class UntrustedModelError(RuntimeError):
    """Raised when attempting to load a model outside trusted directories or invalid format."""


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for an estimator family supported by NETGUARD AI."""

    key: str
    display_name: str
    model_family: Literal["linear", "tree"]
    supports_proba: bool = True


def build_model_pipeline(
    sample_df: pd.DataFrame,
    *,
    estimator: BaseEstimator,
    model_family: Literal["linear", "tree"],
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
) -> Pipeline:
    """Wrap an estimator with a leakage-free preprocessing Pipeline."""
    preprocessor, _, _ = build_full_preprocessing_pipeline(
        sample_df,
        model_family=model_family,
        enable_feature_engineering=enable_feature_engineering,
        task=task,
        sparse_output=True,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def save_model_artifact(
    pipeline: Pipeline,
    destination: Path,
    *,
    compress: int = 3,
) -> Path:
    """Persist a trained Scikit-learn Pipeline safely using Joblib."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination, compress=compress)
    return destination


def load_model_artifact(
    artifact_path: Path,
    *,
    paths: ProjectPaths | None = None,
    allowed_root: Path | None = None,
) -> Pipeline:
    """Safely load a serialized Joblib pipeline after trust and schema checks."""
    resolved_paths = ensure_directories(paths or get_paths())
    trusted_root = (allowed_root or resolved_paths.models_dir).resolve()
    target = artifact_path.resolve()

    if not target.is_file():
        raise FileNotFoundError(f"Model artifact not found: '{artifact_path.name}'.")

    if target.suffix.lower() != ".joblib":
        raise UntrustedModelError(
            f"Refusing to load non-.joblib model artifact: '{target.name}'."
        )

    try:
        target.relative_to(trusted_root)
    except ValueError as exc:
        raise UntrustedModelError(
            f"Refusing to load model outside trusted directory '{trusted_root}'."
        ) from exc

    loaded = joblib.load(target)
    if not isinstance(loaded, Pipeline) or not hasattr(loaded, "predict"):
        raise UntrustedModelError(
            f"Loaded artifact '{target.name}' is not a valid Scikit-learn Pipeline."
        )
    return loaded
