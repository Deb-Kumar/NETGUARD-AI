"""Decision Tree Classifier pipeline and hyperparameter grid."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.base import ModelSpec, build_model_pipeline
from src.utils.config import RANDOM_STATE

MODEL_SPEC = ModelSpec(
    key="decision_tree",
    display_name="Decision Tree Classifier",
    model_family="tree",
    supports_proba=True,
)


def create_decision_tree_pipeline(
    sample_df: pd.DataFrame,
    *,
    criterion: str = "gini",
    max_depth: int | None = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: str | dict[Any, float] | None = "balanced",
    random_state: int = RANDOM_STATE,
    enable_feature_engineering: bool = True,
    task: Literal["binary", "multiclass"] = "binary",
) -> Pipeline:
    """Build a Decision Tree pipeline without unnecessary numerical scaling."""
    estimator = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    return build_model_pipeline(
        sample_df,
        estimator=estimator,
        model_family=MODEL_SPEC.model_family,
        enable_feature_engineering=enable_feature_engineering,
        task=task,
    )


def get_decision_tree_param_grid() -> dict[str, list[Any]]:
    """Return the GridSearchCV parameter grid for Decision Tree Classifier."""
    return {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [12, 20, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 4],
    }
