"""Best model selection using CV validation metrics and single final test evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.data.loader import load_training_and_testing_sets
from src.data.validator import PREDEFINED_FEATURE_COLUMNS, extract_features_and_target
from src.evaluation.evaluate import (
    build_comparison_table,
    evaluate_binary_model,
    save_evaluation_results,
)
from src.evaluation.plots import generate_binary_evaluation_plots
from src.models import MODEL_SPECS, load_model_artifact, save_model_artifact
from src.utils.config import (
    BINARY_TARGET_COLUMN,
    ProjectPaths,
    ensure_directories,
    get_paths,
)

SelectionStrategy = Literal[
    "cybersecurity_composite",
    "highest_recall",
    "highest_f1",
    "lowest_fnr",
]

SELECTION_STRATEGIES: dict[str, str] = {
    "cybersecurity_composite": (
        "Non-redundant weighted cybersecurity score: 0.30*CV_F1 + 0.30*CV_ROC_AUC + "
        "0.25*CV_Recall + 0.15*CV_Precision (computed strictly on validation folds)."
    ),
    "highest_recall": "Maximize attack recall on validation folds (tie-broken by F1).",
    "highest_f1": "Maximize attack F1-score on validation folds (tie-broken by Recall).",
    "lowest_fnr": "Minimize False Negative Rate (1 - Recall) on validation folds.",
}


def compute_selection_score(
    cv_metrics: dict[str, float],
    *,
    strategy: SelectionStrategy = "cybersecurity_composite",
) -> float:
    """Compute selection score solely from validation/cross-validation metrics."""
    recall = float(cv_metrics["recall"])
    precision = float(cv_metrics["precision"])
    f1 = float(cv_metrics["f1"])
    roc_auc = float(cv_metrics.get("roc_auc", f1))
    fnr = 1.0 - recall

    if strategy == "cybersecurity_composite":
        return round(
            0.30 * f1 + 0.30 * roc_auc + 0.25 * recall + 0.15 * precision,
            6,
        )
    if strategy == "highest_recall":
        return round(recall + 0.001 * f1, 6)
    if strategy == "highest_f1":
        return round(f1 + 0.001 * recall, 6)
    if strategy == "lowest_fnr":
        return round((1.0 - fnr) + 0.001 * f1, 6)
    raise ValueError(
        f"Unsupported selection strategy '{strategy}'. "
        f"Choose from {list(SELECTION_STRATEGIES)}."
    )


def _extract_validation_metrics(
    model_key: str,
    *,
    tuning_results: dict[str, Any] | None,
    cv_results: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    """Retrieve validation/CV metrics for a model without touching the test set."""
    if tuning_results and model_key in tuning_results.get("models", {}):
        cv_m = tuning_results["models"][model_key]["best_cv_metrics"]
        return (
            {
                "recall": float(cv_m["recall"]),
                "precision": float(cv_m["precision"]),
                "f1": float(cv_m["f1"]),
                "roc_auc": float(cv_m.get("roc_auc", 0.0)),
                "accuracy": float(cv_m.get("accuracy", 0.0)),
            },
            "GridSearchCV best cross-validation metrics",
        )

    if cv_results and model_key in cv_results.get("models", {}):
        cv_m = cv_results["models"][model_key]["metrics"]
        return (
            {
                "recall": float(cv_m["recall"]["mean"]),
                "precision": float(cv_m["precision"]["mean"]),
                "f1": float(cv_m["f1"]["mean"]),
                "roc_auc": float(cv_m["roc_auc"]["mean"]),
                "accuracy": float(cv_m["accuracy"]["mean"]),
            },
            "Stratified 5-Fold CV metrics",
        )

    raise RuntimeError(
        f"No cross-validation or tuning validation metrics found for '{model_key}'. "
        "Run cross-validation or tuning before model selection so test data is never used for selection."
    )


def select_and_finalize_best_model(
    test_df: pd.DataFrame | None = None,
    *,
    strategy: SelectionStrategy = "cybersecurity_composite",
    cv_results: dict[str, Any] | None = None,
    tuning_results: dict[str, Any] | None = None,
    paths: ProjectPaths | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Select the best model from validation scores, then evaluate once on untouched test set."""
    resolved_paths = ensure_directories(paths or get_paths())

    # Load persisted validation results if not passed explicitly
    if tuning_results is None:
        tuning_file = resolved_paths.model_metadata_dir / "tuning_results.json"
        if tuning_file.is_file():
            tuning_results = json.loads(tuning_file.read_text(encoding="utf-8"))

    if cv_results is None:
        cv_file = resolved_paths.model_metadata_dir / "cv_results.json"
        if cv_file.is_file():
            cv_results = json.loads(cv_file.read_text(encoding="utf-8"))

    # 1. Rank candidates strictly on validation/CV performance (never test set!)
    validation_ranking: list[dict[str, Any]] = []
    loaded_models: dict[str, Any] = {}

    for model_key, spec in MODEL_SPECS.items():
        artifact_path = resolved_paths.binary_models_dir / f"{model_key}.joblib"
        if not artifact_path.is_file():
            continue
        loaded_models[model_key] = load_model_artifact(artifact_path, paths=resolved_paths)
        val_metrics, val_source = _extract_validation_metrics(
            model_key, tuning_results=tuning_results, cv_results=cv_results
        )
        score = compute_selection_score(val_metrics, strategy=strategy)
        validation_ranking.append(
            {
                "model_key": model_key,
                "model_name": spec.display_name,
                "validation_source": val_source,
                "validation_metrics": val_metrics,
                "validation_fnr": round(1.0 - val_metrics["recall"], 4),
                "selection_score": score,
            }
        )

    if not validation_ranking:
        raise RuntimeError(
            "No trained binary model artifacts found in models/binary/. Run training first."
        )

    validation_ranking.sort(key=lambda r: r["selection_score"], reverse=True)
    winner = validation_ranking[0]
    winner_key = winner["model_key"]
    winner_pipeline = loaded_models[winner_key]

    # Save winner as models/binary/best_model.joblib
    best_model_path = resolved_paths.binary_models_dir / "best_model.joblib"
    save_model_artifact(winner_pipeline, best_model_path)

    # 2. Single final evaluation across all models on untouched test set
    if test_df is None:
        _, test_df = load_training_and_testing_sets(paths=resolved_paths)
    X_test, y_test = extract_features_and_target(test_df, task="binary")

    test_evaluations: dict[str, dict[str, Any]] = {}
    for key, pipeline in loaded_models.items():
        spec = MODEL_SPECS[key]
        test_evaluations[key] = evaluate_binary_model(
            pipeline,
            X_test,
            y_test,
            model_key=key,
            model_name=spec.display_name,
            split_name="untouched_test",
        )

    eval_path = save_evaluation_results(test_evaluations, paths=resolved_paths)
    plot_paths = (
        generate_binary_evaluation_plots(test_evaluations, paths=resolved_paths)
        if generate_plots
        else {}
    )

    prep_config_path = resolved_paths.model_metadata_dir / "preprocessing_config.json"
    prep_config = (
        json.loads(prep_config_path.read_text(encoding="utf-8"))
        if prep_config_path.is_file()
        else {}
    )

    winner_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in winner_pipeline.named_steps["classifier"].get_params().items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }

    metadata_payload: dict[str, Any] = {
        "project": "NETGUARD AI",
        "task": "binary",
        "target_name": BINARY_TARGET_COLUMN,
        "class_mapping": {"0": "Normal Traffic", "1": "Malicious Traffic"},
        "dataset_version": "UNSW-NB15 predefined training/testing sets",
        "feature_names": list(PREDEFINED_FEATURE_COLUMNS),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "selection_strategy": strategy,
        "selection_strategy_description": SELECTION_STRATEGIES[strategy],
        "selected_model_key": winner_key,
        "selected_model_name": winner["model_name"],
        "best_model_artifact": str(best_model_path),
        "validation_ranking": validation_ranking,
        "selected_validation_metrics": winner["validation_metrics"],
        "final_test_evaluation": test_evaluations[winner_key],
        "all_models_test_comparison": build_comparison_table(
            list(test_evaluations.values())
        ).to_dict(orient="records"),
        "evaluation_results_path": str(eval_path),
        "evaluation_plots": plot_paths,
        "model_parameters": winner_params,
        "preprocessing_configuration": prep_config,
    }

    metadata_path = resolved_paths.model_metadata_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    metadata_payload["metadata_path"] = str(metadata_path)
    return metadata_payload
