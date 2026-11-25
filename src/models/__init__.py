"""Model registry and shared persistence exports for NETGUARD AI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.pipeline import Pipeline

from src.models.base import (
    ModelSpec,
    UntrustedModelError,
    build_model_pipeline,
    load_model_artifact,
    save_model_artifact,
)
from src.models.decision_tree import (
    MODEL_SPEC as DT_SPEC,
    create_decision_tree_pipeline,
    get_decision_tree_param_grid,
)
from src.models.logistic_regression import (
    MODEL_SPEC as LR_SPEC,
    create_logistic_regression_pipeline,
    get_logistic_regression_param_grid,
)
from src.models.random_forest import (
    MODEL_SPEC as RF_SPEC,
    create_random_forest_pipeline,
    get_random_forest_param_grid,
)

MODEL_SPECS: dict[str, ModelSpec] = {
    LR_SPEC.key: LR_SPEC,
    DT_SPEC.key: DT_SPEC,
    RF_SPEC.key: RF_SPEC,
}

MODEL_BUILDERS: dict[str, Callable[..., Pipeline]] = {
    LR_SPEC.key: create_logistic_regression_pipeline,
    DT_SPEC.key: create_decision_tree_pipeline,
    RF_SPEC.key: create_random_forest_pipeline,
}

PARAM_GRID_BUILDERS: dict[str, Callable[[], dict[str, list[Any]]]] = {
    LR_SPEC.key: get_logistic_regression_param_grid,
    DT_SPEC.key: get_decision_tree_param_grid,
    RF_SPEC.key: get_random_forest_param_grid,
}

__all__ = [
    "MODEL_BUILDERS",
    "MODEL_SPECS",
    "ModelSpec",
    "PARAM_GRID_BUILDERS",
    "UntrustedModelError",
    "build_model_pipeline",
    "create_decision_tree_pipeline",
    "create_logistic_regression_pipeline",
    "create_random_forest_pipeline",
    "get_decision_tree_param_grid",
    "get_logistic_regression_param_grid",
    "get_random_forest_param_grid",
    "load_model_artifact",
    "save_model_artifact",
]
