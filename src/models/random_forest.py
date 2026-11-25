"""Random Forest Classifier pipeline and hyperparameter grid."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.base import ModelSpec, build_model_pipeline
from src.utils.config import DEFAULT_N_JOBS, RANDOM_STATE

MODEL_SPEC = ModelSpec(
    key="random_forest",
    display_name="Random Forest Classifier",
    model_family="tree",
    supports_proba=True,
)


def create_random_forest_pipeline(
    sample_df: pd.DataFrame,
    *,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: str | dict[Any, float] | None = "balanced",
    n_jobs: int = DEFAULT_N_JOBS,
    random_state: int = RANDOM_STATE,
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
) -> Pipeline:
    """Build a Random Forest pipeline with bounded worker concurrency."""
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return build_model_pipeline(
        sample_df,
        estimator=estimator,
        model_family=MODEL_SPEC.model_family,
        enable_feature_engineering=enable_feature_engineering,
        task=task,
    )


def get_random_forest_param_grid() -> dict[str, list[Any]]:
    """Return a manageable GridSearchCV parameter grid for Random Forest."""
    return {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [15, 25],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 4],
    }
