"""Phase 3 tests: dataset loading, profiling, schema validation, and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main import main as cli_main
from src.data import (
    DataLeakageError,
    DatasetLoadError,
    PREDEFINED_COLUMNS,
    PREDEFINED_FEATURE_COLUMNS,
    RAW_49_COLUMNS,
    extract_features_and_target,
    get_feature_metadata_map,
    iter_csv_chunks,
    load_csv_dataset,
    load_feature_descriptions,
    load_raw_part,
    load_training_and_testing_sets,
    profile_dataset,
    validate_no_target_leakage,
    validate_predefined_dataset,
    validate_train_test_compatibility,
)
from src.utils.config import get_paths


def _make_valid_predefined_df(rows: int = 4) -> pd.DataFrame:
    """Create a small synthetic DataFrame conforming to the 45-column schema."""
    data: dict[str, list[object]] = {}
    for col in PREDEFINED_COLUMNS:
        if col == "id":
            data[col] = list(range(1, rows + 1))
        elif col == "proto":
            data[col] = ["tcp" if i % 2 == 0 else "udp" for i in range(rows)]
        elif col == "service":
            data[col] = ["-" if i % 2 == 0 else "http" for i in range(rows)]
        elif col == "state":
            data[col] = ["FIN" if i % 2 == 0 else "INT" for i in range(rows)]
        elif col == "attack_cat":
            data[col] = ["Normal" if i % 2 == 0 else "Exploits" for i in range(rows)]
        elif col == "label":
            data[col] = [0 if i % 2 == 0 else 1 for i in range(rows)]
        else:
            data[col] = [float(i + 1) for i in range(rows)]
    return pd.DataFrame(data)


def test_load_feature_descriptions_and_metadata_map() -> None:
    feat_df = load_feature_descriptions()
    assert len(feat_df) == 49
    assert {"No.", "Name", "Type", "Description", "normalized_name", "predefined_name"}.issubset(
        feat_df.columns
    )

    metadata_map = get_feature_metadata_map()
    for col in PREDEFINED_COLUMNS:
        assert col in metadata_map, f"Missing description for column '{col}'"
        assert metadata_map[col]["description"]


def test_load_training_and_testing_sets_full_and_profile() -> None:
    train_df, test_df = load_training_and_testing_sets()
    assert train_df.shape == (175341, 45)
    assert test_df.shape == (82332, 45)
    assert list(train_df.columns) == list(PREDEFINED_COLUMNS)
    assert list(test_df.columns) == list(PREDEFINED_COLUMNS)

    train_profile = profile_dataset(train_df, name="train")
    assert train_profile["num_records"] == 175341
    assert train_profile["num_columns"] == 45
    assert train_profile["missing_values"]["total_missing_cells"] == 0
    assert train_profile["dash_placeholders"]["service"] == 94168
    assert train_profile["target_distribution"]["binary_counts"] == {"1": 119341, "0": 56000}

    validation = validate_train_test_compatibility(train_df, test_df)
    assert validation.is_valid is True
    assert validation.errors == []
    assert validation.summary["unseen_test_categories"]["state"] == ["ACC", "CLO"]


def test_load_training_downcast_reduces_memory() -> None:
    full_df, _ = load_training_and_testing_sets(nrows_train=5000, nrows_test=100)
    slim_df, _ = load_training_and_testing_sets(
        nrows_train=5000,
        nrows_test=100,
        downcast_numeric=True,
    )
    assert slim_df.memory_usage(deep=True).sum() < full_df.memory_usage(deep=True).sum()


def test_load_raw_part_and_chunked_streaming() -> None:
    part1_sample = load_raw_part(1, nrows=500)
    assert part1_sample.shape == (500, 49)
    assert list(part1_sample.columns) == list(RAW_49_COLUMNS)
    assert part1_sample["attack_cat"].isna().sum() == 0

    paths = get_paths()
    part1_path = paths.raw_part_paths["UNSW-NB15_1.csv"]
    assert part1_path is not None
    chunks = iter_csv_chunks(part1_path, chunksize=250, has_header=False, column_names=RAW_49_COLUMNS)
    first_chunk = next(chunks)
    second_chunk = next(chunks)
    assert len(first_chunk) == 250
    assert len(second_chunk) == 250


def test_loader_error_handling_for_missing_empty_and_invalid(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetLoadError):
        load_csv_dataset(missing_file)

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_csv_dataset(empty_file)

    with pytest.raises(ValueError):
        load_raw_part(5)

    with pytest.raises(ValueError):
        next(iter_csv_chunks(empty_file, chunksize=0))


def test_validator_catches_leakage_and_invalid_targets() -> None:
    with pytest.raises(DataLeakageError):
        validate_no_target_leakage(["dur", "attack_cat"], task="binary")

    with pytest.raises(DataLeakageError):
        validate_no_target_leakage(["dur", "label"], task="multiclass")

    with pytest.raises(DataLeakageError):
        validate_no_target_leakage(["id", "dur"], task="binary")

    df = _make_valid_predefined_df(4)
    assert validate_predefined_dataset(df).is_valid is True

    X_bin, y_bin = extract_features_and_target(df, task="binary")
    assert list(X_bin.columns) == list(PREDEFINED_FEATURE_COLUMNS)
    assert "attack_cat" not in X_bin.columns
    assert "label" not in X_bin.columns
    assert "id" not in X_bin.columns
    assert list(y_bin) == [0, 1, 0, 1]

    X_multi, y_multi = extract_features_and_target(df, task="multiclass")
    assert list(X_multi.columns) == list(PREDEFINED_FEATURE_COLUMNS)
    assert list(y_multi) == ["Normal", "Exploits", "Normal", "Exploits"]

    # Corrupt label/attack_cat consistency
    bad_df = df.copy()
    bad_df.loc[0, "attack_cat"] = "DoS"
    res = validate_predefined_dataset(bad_df)
    assert res.is_valid is False
    assert any("consistency" in err for err in res.errors)

    # Corrupt infinite values
    inf_df = df.copy()
    inf_df.loc[1, "dur"] = np.inf
    res_inf = validate_predefined_dataset(inf_df)
    assert res_inf.is_valid is False
    assert any("Infinite" in err for err in res_inf.errors)


def test_validator_supports_prediction_only_dataframe() -> None:
    df = _make_valid_predefined_df(4).drop(columns=["id", "attack_cat", "label"])
    res = validate_predefined_dataset(df, dataset_name="predict_input", require_targets=False)
    assert res.is_valid is True


def test_cli_inspect_data_saves_report_and_passes(monkeypatch, capsys) -> None:
    monkeypatch.setenv("NETGUARD_PROJECT_ROOT", str(get_paths().root))
    exit_code = cli_main(["inspect-data", "--downcast", "--save-report"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["training_set"]["num_records"] == 175341
    assert payload["testing_set"]["num_records"] == 82332
    assert payload["validation"]["is_valid"] is True
    assert Path(payload["saved_report_path"]).is_file()
