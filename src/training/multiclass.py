"""Independent multiclass attack-category (attack_cat) classification pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.loader import load_training_and_testing_sets
from src.data.validator import PREDEFINED_FEATURE_COLUMNS, extract_features_and_target
from src.evaluation.evaluate import evaluate_multiclass_model
from src.evaluation.plots import plot_multiclass_confusion_matrix
from src.models import MODEL_BUILDERS, MODEL_SPECS, save_model_artifact
from src.utils.config import (
    MULTICLASS_TARGET_COLUMN,
    ProjectPaths,
    ensure_directories,
    get_paths,
)


def train_and_evaluate_multiclass(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    model_keys: tuple[str, ...] = ("decision_tree", "random_forest"),
    enable_feature_engineering: bool = True,
    paths: ProjectPaths | None = None,
    generate_plot: bool = True,
) -> dict[str, Any]:
    """Train and evaluate independent multiclass classifiers for attack_cat.

    Strictly excludes binary 'label' and 'id' from input features and uses a
    separate pipeline directory (models/multiclass/).
    """
    resolved_paths = ensure_directories(paths or get_paths())
    if train_df is None or test_df is None:
        loaded_tr, loaded_te = load_training_and_testing_sets(paths=resolved_paths)
        train_df = loaded_tr if train_df is None else train_df
        test_df = loaded_te if test_df is None else test_df

    X_train, y_train_raw = extract_features_and_target(train_df, task="multiclass")
    X_test, y_test_raw = extract_features_and_target(test_df, task="multiclass")
    y_train = y_train_raw.astype(str).str.strip()
    y_test = y_test_raw.astype(str).str.strip()

    discovered_classes = sorted(y_train.unique().tolist())

    candidates: dict[str, dict[str, Any]] = {}
    pipelines: dict[str, Any] = {}

    for key in model_keys:
        if key not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model key '{key}'.")
        spec = MODEL_SPECS[key]
        pipe = MODEL_BUILDERS[key](
            X_train,
            class_weight="balanced",
            enable_feature_engineering=enable_feature_engineering,
            task="multiclass",
        )
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_sec = time.perf_counter() - t0

        ev = evaluate_multiclass_model(
            pipe,
            X_test,
            y_test,
            model_key=key,
            model_name=spec.display_name,
            split_name="untouched_test",
        )
        ev["training_duration_seconds"] = round(train_sec, 3)
        candidates[key] = ev
        pipelines[key] = pipe

        model_out = resolved_paths.multiclass_models_dir / f"{key}.joblib"
        save_model_artifact(pipe, model_out)

    # Select winner by macro F1 (balances rare classes like Worms/Shellcode with Normal/Generic)
    winner_key = max(
        candidates.keys(),
        key=lambda k: (
            candidates[k]["metrics"]["macro_f1"],
            candidates[k]["metrics"]["weighted_f1"],
        ),
    )
    best_pipe = pipelines[winner_key]
    best_path = resolved_paths.multiclass_models_dir / "best_model.joblib"
    save_model_artifact(best_pipe, best_path)

    plot_path: str | None = None
    if generate_plot:
        fig_out = resolved_paths.reports_figures_dir / "13_multiclass_confusion_matrix.png"
        plot_multiclass_confusion_matrix(candidates[winner_key], fig_out)
        plot_path = str(fig_out)

    metadata = {
        "task": "multiclass",
        "target_name": MULTICLASS_TARGET_COLUMN,
        "independent_from_binary": True,
        "excluded_columns": ["id", "label", "attack_cat"],
        "feature_names": list(PREDEFINED_FEATURE_COLUMNS),
        "discovered_classes": discovered_classes,
        "normal_class_label": "Normal",
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model_key": winner_key,
        "selected_model_name": MODEL_SPECS[winner_key].display_name,
        "best_model_artifact": str(best_path),
        "confusion_matrix_plot": plot_path,
        "models": candidates,
    }

    meta_path = resolved_paths.model_metadata_dir / "multiclass_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path)
    return metadata
