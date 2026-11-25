"""Logistic Regression model pipeline and hyperparameter grid."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.base import ModelSpec, build_model_pipeline
from src.utils.config import RANDOM_STATE

MODEL_SPEC = ModelSpec(
    key="logistic_regression",
    display_name="Logistic Regression",
    model_family="linear",
    supports_proba=True,
)


def create_logistic_regression_pipeline(
    sample_df: pd.DataFrame,
    *,
    C: float = 1.0,
    solver: str = "lbfgs",
    class_weight: str | dict[Any, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
) -> Pipeline:
    """Build a Logistic Regression pipeline with scaled numeric & OHE categorical features."""
    estimator = LogisticRegression(
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    return build_model_pipeline(
        sample_df,
        estimator=estimator,
        model_family=MODEL_SPEC.model_family,
        enable_feature_engineering=enable_feature_engineering,
        task=task,
    )


def get_logistic_regression_param_grid() -> dict[str, list[Any]]:
    """Return the GridSearchCV parameter grid for Logistic Regression."""
    return {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__class_weight": [None, "balanced"],
    }
