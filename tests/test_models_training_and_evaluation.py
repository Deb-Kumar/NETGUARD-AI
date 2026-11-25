"""Phase 6-10 tests: training, CV, GridSearchCV tuning, selection, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import PREDEFINED_COLUMNS
from src.evaluation import build_comparison_table, evaluate_binary_model
from src.models import UntrustedModelError, load_model_artifact
from src.training import (
    run_grid_search_tuning,
    run_stratified_cv,
    select_and_finalize_best_model,
    train_all_binary_models,
)
from src.utils.config import get_paths


def _make_synthetic_dataset(rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list[object]] = {}
    labels = [0 if i % 2 == 0 else 1 for i in range(rows)]

    for col in PREDEFINED_COLUMNS:
        if col == "id":
            data[col] = list(range(1, rows + 1))
        elif col == "proto":
            data[col] = ["tcp" if lbl == 0 else "udp" for lbl in labels]
        elif col == "service":
            data[col] = ["http" if i % 3 == 0 else "-" for i in range(rows)]
        elif col == "state":
            data[col] = ["FIN" if lbl == 0 else "INT" for lbl in labels]
        elif col == "attack_cat":
            data[col] = ["Normal" if lbl == 0 else "Exploits" for lbl in labels]
        elif col == "label":
            data[col] = labels
        elif col == "sttl":
            data[col] = [
                int(31 + rng.integers(0, 5)) if lbl == 0 else int(254 - rng.integers(0, 5))
                for lbl in labels
            ]
        else:
            data[col] = [
                float(rng.normal(10.0 if lbl == 0 else 50.0, 3.0)) for lbl in labels
            ]
    return pd.DataFrame(data)


def test_end_to_end_train_cv_tune_select_and_evaluate(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    train_df = _make_synthetic_dataset(rows=80, seed=42)
    test_df = _make_synthetic_dataset(rows=40, seed=99)
    # Introduce unseen categorical value in test set to verify handle_unknown='ignore'
    test_df.loc[0, "state"] = "ACC"

    train_summary = train_all_binary_models(
        train_df=train_df,
        run_imbalance_study=False,
        paths=temp_paths,
    )
    assert Path(train_summary["summary_path"]).is_file()
    assert Path(train_summary["preprocessing_config_path"]).is_file()
    assert set(train_summary["models"].keys()) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }

    cv_summary = run_stratified_cv(
        train_df=train_df,
        n_splits=3,
        paths=temp_paths,
    )
    assert Path(cv_summary["results_path"]).is_file()
    assert len(cv_summary["models"]["logistic_regression"]["metrics"]["recall"]["fold_scores"]) == 3

    tiny_grids = {
        "logistic_regression": {"classifier__C": [0.5, 1.0], "classifier__solver": ["lbfgs"]},
        "decision_tree": {"classifier__max_depth": [3, 6]},
        "random_forest": {"classifier__n_estimators": [10, 20], "classifier__max_depth": [5]},
    }
    tuning_summary = run_grid_search_tuning(
        train_df=train_df,
        cv_folds=3,
        scoring="f1",
        custom_grids=tiny_grids,
        max_tuning_samples=60,
        refit_full_train=True,
        paths=temp_paths,
    )
    assert Path(tuning_summary["results_path"]).is_file()
    assert tuning_summary["final_refit_samples"] == 80

    metadata_payload = select_and_finalize_best_model(
        test_df=test_df,
        strategy="cybersecurity_composite",
        paths=temp_paths,
        generate_plots=True,
    )
    assert Path(metadata_payload["best_model_artifact"]).is_file()
    assert Path(metadata_payload["metadata_path"]).is_file()
    assert len(metadata_payload["evaluation_plots"]) == 4
    for fig_path in metadata_payload["evaluation_plots"].values():
        assert Path(fig_path).is_file()

    best_model = load_model_artifact(
        Path(metadata_payload["best_model_artifact"]),
        paths=temp_paths,
    )
    X_test = test_df.drop(columns=["id", "attack_cat", "label"])
    y_test = test_df["label"]
    ev = evaluate_binary_model(
        best_model,
        X_test,
        y_test,
        model_key="best",
        model_name="Best",
    )
    assert 0.0 <= ev["metrics"]["recall"] <= 1.0
    assert 0.0 <= ev["metrics"]["false_negative_rate"] <= 1.0
    comp = build_comparison_table([ev])
    assert len(comp) == 1


def test_load_model_artifact_blocks_untrusted_paths(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    outside_file = tmp_path / "evil.joblib"
    outside_file.write_text("not a real joblib", encoding="utf-8")
    with pytest.raises(UntrustedModelError):
        load_model_artifact(outside_file, paths=temp_paths)

    bad_ext = temp_paths.binary_models_dir / "model.pkl"
    bad_ext.parent.mkdir(parents=True, exist_ok=True)
    bad_ext.write_text("bad", encoding="utf-8")
    with pytest.raises(UntrustedModelError):
        load_model_artifact(bad_ext, paths=temp_paths)
