"""Phase 5 tests: preprocessing pipelines, unseen categories, and leakage prevention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataLeakageError, PREDEFINED_COLUMNS
from src.preprocessing import (
    ENGINEERED_FEATURE_NAMES,
    NetworkFeatureEngineer,
    analyze_class_imbalance,
    build_full_preprocessing_pipeline,
    save_preprocessing_config,
    split_feature_columns,
)
from src.utils.config import get_paths


def _make_train_test_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 12
    train_data: dict[str, list[object]] = {}
    for col in PREDEFINED_COLUMNS:
        if col == "id":
            train_data[col] = list(range(1, rows + 1))
        elif col == "proto":
            train_data[col] = ["tcp" if i % 2 == 0 else "udp" for i in range(rows)]
        elif col == "service":
            train_data[col] = ["-" if i % 3 == 0 else "http" for i in range(rows)]
        elif col == "state":
            train_data[col] = ["FIN" if i % 2 == 0 else "INT" for i in range(rows)]
        elif col == "attack_cat":
            train_data[col] = ["Normal" if i % 3 == 0 else "Exploits" for i in range(rows)]
        elif col == "label":
            train_data[col] = [0 if i % 3 == 0 else 1 for i in range(rows)]
        else:
            train_data[col] = [float(i + 1) for i in range(rows)]

    train_df = pd.DataFrame(train_data)
    test_df = train_df.iloc[:4].copy()
    test_df.loc[test_df.index[0], "state"] = "ACC"  # Unseen category in test
    test_df.loc[test_df.index[1], "dur"] = np.nan  # Missing numeric value in test
    test_df.loc[test_df.index[2], "service"] = None  # Missing categorical value in test
    return train_df, test_df


def test_split_feature_columns_blocks_leakage() -> None:
    train_df, _ = _make_train_test_sample()
    num_cols, cat_cols = split_feature_columns(train_df)
    assert "id" not in num_cols and "id" not in cat_cols
    assert "label" not in num_cols and "label" not in cat_cols
    assert "attack_cat" not in num_cols and "attack_cat" not in cat_cols
    assert set(cat_cols) == {"proto", "service", "state"}

    with pytest.raises(DataLeakageError):
        split_feature_columns(train_df, feature_columns=["dur", "attack_cat"], task="binary")


def test_feature_engineer_adds_expected_columns_safely() -> None:
    train_df, _ = _make_train_test_sample()
    features_only = train_df.drop(columns=["id", "attack_cat", "label"])
    engineer = NetworkFeatureEngineer(enabled=True)
    transformed = engineer.fit_transform(features_only)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in transformed.columns
    assert transformed["is_service_unspecified"].iloc[0] == 1


def test_linear_and_tree_pipelines_handle_unseen_and_missing_values() -> None:
    train_df, test_df = _make_train_test_sample()
    X_train = train_df.drop(columns=["id", "attack_cat", "label"])
    X_test = test_df.drop(columns=["id", "attack_cat", "label"])

    linear_pipe, num_cols, cat_cols = build_full_preprocessing_pipeline(
        X_train, model_family="linear", enable_feature_engineering=True
    )
    tree_pipe, _, _ = build_full_preprocessing_pipeline(
        X_train, model_family="tree", enable_feature_engineering=True
    )

    assert len(num_cols) > 39
    assert set(cat_cols) == {"proto", "service", "state"}

    X_train_lin = linear_pipe.fit_transform(X_train)
    X_test_lin = linear_pipe.transform(X_test)
    assert X_train_lin.shape[1] == X_test_lin.shape[1]

    X_train_tree = tree_pipe.fit_transform(X_train)
    X_test_tree = tree_pipe.transform(X_test)
    assert X_train_tree.shape[1] == X_test_tree.shape[1]


def test_imbalance_analysis_and_config_save(tmp_path: Path) -> None:
    train_df, _ = _make_train_test_sample()
    imbalance = analyze_class_imbalance(train_df["label"])
    assert imbalance["total_samples"] == 12
    assert set(imbalance["class_counts"].keys()) == {"0", "1"}

    temp_paths = get_paths(tmp_path)
    out = save_preprocessing_config(
        numerical_columns=["dur", "sbytes"],
        categorical_columns=["proto", "service", "state"],
        imbalance_report=imbalance,
        paths=temp_paths,
    )
    assert out.is_file()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["excluded_columns"] == ["id", "attack_cat", "label"]
