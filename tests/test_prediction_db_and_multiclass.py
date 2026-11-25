"""Phase 11, 13 & 14 tests: prediction engine, SQLite history, and multiclass pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import PREDEFINED_COLUMNS, PREDEFINED_FEATURE_COLUMNS
from src.database import clear_history, export_history_csv, init_db, log_prediction, query_history
from src.prediction import (
    PredictionValidationError,
    predict_batch_csv,
    predict_single_record,
)
from src.training import (
    run_stratified_cv,
    select_and_finalize_best_model,
    train_all_binary_models,
    train_and_evaluate_multiclass,
)
from src.utils.config import get_paths


def _make_multiclass_dataset(rows: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = ["Normal", "Generic", "Exploits"]
    data: dict[str, list[object]] = {}
    assigned_cats = [cats[i % len(cats)] for i in range(rows)]
    labels = [0 if c == "Normal" else 1 for c in assigned_cats]

    for col in PREDEFINED_COLUMNS:
        if col == "id":
            data[col] = list(range(1, rows + 1))
        elif col == "proto":
            data[col] = ["tcp" if c != "Generic" else "udp" for c in assigned_cats]
        elif col == "service":
            data[col] = ["http" if c == "Exploits" else "dns" if c == "Generic" else "-" for c in assigned_cats]
        elif col == "state":
            data[col] = ["FIN" if c == "Normal" else "INT" for c in assigned_cats]
        elif col == "attack_cat":
            data[col] = assigned_cats
        elif col == "label":
            data[col] = labels
        elif col == "sttl":
            data[col] = [
                31 if c == "Normal" else (254 if c == "Exploits" else 62)
                for c in assigned_cats
            ]
        else:
            offset = {"Normal": 5.0, "Generic": 25.0, "Exploits": 60.0}
            data[col] = [float(max(0.0, rng.normal(offset[c], 2.0))) for c in assigned_cats]
    return pd.DataFrame(data)


def test_single_batch_prediction_and_multiclass_and_db_history(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    train_df = _make_multiclass_dataset(rows=90, seed=7)
    test_df = _make_multiclass_dataset(rows=30, seed=19)

    train_all_binary_models(train_df=train_df, run_imbalance_study=False, paths=temp_paths)
    run_stratified_cv(train_df=train_df, n_splits=3, paths=temp_paths)
    select_and_finalize_best_model(test_df=test_df, paths=temp_paths, generate_plots=False)

    mc_meta = train_and_evaluate_multiclass(
        train_df=train_df,
        test_df=test_df,
        model_keys=("decision_tree", "random_forest"),
        paths=temp_paths,
        generate_plot=True,
    )
    assert Path(mc_meta["best_model_artifact"]).is_file()
    assert Path(mc_meta["metadata_path"]).is_file()
    assert Path(mc_meta["confusion_matrix_plot"]).is_file()
    assert mc_meta["independent_from_binary"] is True
    assert mc_meta["discovered_classes"] == ["Exploits", "Generic", "Normal"]

    # Single-record prediction (shuffled feature key order)
    sample_row = test_df[list(PREDEFINED_FEATURE_COLUMNS)].iloc[0].to_dict()
    single_res = predict_single_record(sample_row, paths=temp_paths)
    assert single_res.predicted_label in (0, 1)
    assert single_res.malicious_probability is not None
    assert single_res.multiclass_attack_category in ("Normal", "Generic", "Exploits")

    # Batch CSV without label (prediction-only)
    pred_only_csv = tmp_path / "batch_no_label.csv"
    test_df[list(PREDEFINED_FEATURE_COLUMNS)].to_csv(pred_only_csv, index=False)
    batch_res = predict_batch_csv(pred_only_csv, paths=temp_paths)
    assert batch_res.total_records == 30
    assert batch_res.normal_count + batch_res.malicious_count == 30
    assert batch_res.has_ground_truth is False
    assert batch_res.ground_truth_metrics is None

    # Batch CSV with label (optional evaluation enabled)
    gt_csv = tmp_path / "batch_with_label.csv"
    test_df.to_csv(gt_csv, index=False)
    gt_res = predict_batch_csv(gt_csv, paths=temp_paths)
    assert gt_res.has_ground_truth is True
    assert gt_res.ground_truth_metrics is not None
    assert "accuracy" in gt_res.ground_truth_metrics["metrics"]

    # SQLite prediction history lifecycle
    db_path = init_db(paths=temp_paths)
    assert db_path.is_file()
    row_id = log_prediction(
        prediction_type="Single",
        model_name=single_res.model_name,
        predicted_class=single_res.predicted_class,
        probability=single_res.malicious_probability,
        record_reference="flow-001",
        paths=temp_paths,
    )
    assert row_id >= 1
    log_prediction(
        prediction_type="Batch",
        model_name=batch_res.model_name,
        predicted_class="Malicious Traffic",
        probability=0.88,
        record_reference="batch_no_label.csv",
        paths=temp_paths,
    )

    all_hist = query_history(paths=temp_paths)
    assert len(all_hist) == 2
    batch_only = query_history(prediction_type="Batch", search_text="batch_no_label", paths=temp_paths)
    assert len(batch_only) == 1
    csv_export = export_history_csv(paths=temp_paths)
    assert isinstance(csv_export, str) and "flow-001" in csv_export

    with pytest.raises(ValueError):
        clear_history(confirm=False, paths=temp_paths)
    deleted = clear_history(confirm=True, paths=temp_paths)
    assert deleted == 2
    assert query_history(paths=temp_paths).empty


def test_prediction_validation_rejects_bad_inputs(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    train_df = _make_multiclass_dataset(rows=60, seed=3)
    train_all_binary_models(train_df=train_df, run_imbalance_study=False, paths=temp_paths)
    run_stratified_cv(train_df=train_df, n_splits=3, paths=temp_paths)
    select_and_finalize_best_model(test_df=train_df.iloc[:15], paths=temp_paths, generate_plots=False)

    with pytest.raises(PredictionValidationError):
        predict_single_record({}, paths=temp_paths)

    with pytest.raises(PredictionValidationError):
        predict_single_record({"dur": 0.1}, paths=temp_paths)

    bad_ext = tmp_path / "malicious.exe"
    bad_ext.write_text("bad", encoding="utf-8")
    with pytest.raises(PredictionValidationError):
        predict_batch_csv(bad_ext, paths=temp_paths)

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(PredictionValidationError):
        predict_batch_csv(empty_csv, paths=temp_paths)

    missing_cols_csv = tmp_path / "missing_cols.csv"
    pd.DataFrame({"dur": [1.0, 2.0]}).to_csv(missing_cols_csv, index=False)
    with pytest.raises(PredictionValidationError):
        predict_batch_csv(missing_cols_csv, paths=temp_paths)
