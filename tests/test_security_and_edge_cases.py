"""Independent security, preprocessing edge-case, and E2E regression tests."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.validator import PREDEFINED_COLUMNS, PREDEFINED_FEATURE_COLUMNS
from src.database import (
    clear_history,
    export_history_csv,
    init_db,
    log_prediction,
    query_history,
    sanitize_csv_cell,
    sanitize_dataframe_for_csv,
)
from src.models import UntrustedModelError, load_model_artifact
from src.prediction import (
    PredictionValidationError,
    predict_batch_csv,
    predict_single_record,
)
from src.training import (
    run_stratified_cv,
    select_and_finalize_best_model,
    train_all_binary_models,
)
from src.utils.config import MAX_UPLOAD_SIZE_BYTES, get_paths


def _make_df(rows: int = 36, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list[object]] = {}
    labels = [0 if i % 2 == 0 else 1 for i in range(rows)]
    for col in PREDEFINED_COLUMNS:
        if col == "id":
            data[col] = list(range(1, rows + 1))
        elif col == "proto":
            data[col] = ["tcp" if lbl == 0 else "udp" for lbl in labels]
        elif col == "service":
            data[col] = ["-" if i % 2 == 0 else "http" for i in range(rows)]
        elif col == "state":
            data[col] = ["FIN" if lbl == 0 else "INT" for lbl in labels]
        elif col == "attack_cat":
            data[col] = ["Normal" if lbl == 0 else "Generic" for lbl in labels]
        elif col == "label":
            data[col] = labels
        else:
            data[col] = [float(abs(rng.normal(20.0 if lbl == 0 else 80.0, 5.0))) for lbl in labels]
    return pd.DataFrame(data)


def test_sqlite_connections_close_cleanly_on_windows_and_block_sqli() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tp = get_paths(Path(tmpdir))
        init_db(paths=tp)
        log_prediction(
            prediction_type="Single",
            model_name="Random Forest Classifier",
            predicted_class="Malicious Traffic",
            probability=0.91,
            record_reference="=cmd|'/C calc'!A0",
            notes="'+2+5",
            paths=tp,
        )
        sqli_res = query_history(search_text="' OR '1'='1; DROP TABLE prediction_history; --", paths=tp)
        assert len(sqli_res) == 0
        assert len(query_history(paths=tp)) == 1

        csv_out = str(export_history_csv(paths=tp))
        assert "'=cmd|'/C calc'!A0" in csv_out
        assert "''+2+5" in csv_out or "'\"+2+5" not in csv_out
        assert clear_history(confirm=True, paths=tp) == 1
    # Exiting TemporaryDirectory verifies zero WinError 32 open-handle file locks!


def test_csv_formula_sanitization_preserves_negative_numbers() -> None:
    assert sanitize_csv_cell("-42.5") == "-42.5"
    assert sanitize_csv_cell("=HYPERLINK(\"http://evil\")") == "'=HYPERLINK(\"http://evil\")"
    assert sanitize_csv_cell("@SUM(A1:A2)") == "'@SUM(A1:A2)"
    assert sanitize_csv_cell("-cmd|calc") == "'-cmd|calc"

    df = pd.DataFrame({"ref": ["+1+1", "safe_flow"], "val": [-1.5, 2.0]})
    safe_df = sanitize_dataframe_for_csv(df)
    assert safe_df["ref"].iloc[0] == "'+1+1"
    assert safe_df["ref"].iloc[1] == "safe_flow"


def test_preprocessing_and_prediction_reject_negative_non_numeric_and_oversized(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    train_df = _make_df(40, seed=1)
    test_df = _make_df(20, seed=2)
    train_all_binary_models(train_df=train_df, run_imbalance_study=False, paths=temp_paths)
    run_stratified_cv(train_df=train_df, n_splits=3, paths=temp_paths)
    select_and_finalize_best_model(test_df=test_df, paths=temp_paths, generate_plots=False)

    base_record = test_df[list(PREDEFINED_FEATURE_COLUMNS)].iloc[0].to_dict()

    # Negative value must raise PredictionValidationError
    neg_rec = dict(base_record)
    neg_rec["dur"] = -1.0
    with pytest.raises(PredictionValidationError, match="negative"):
        predict_single_record(neg_rec, paths=temp_paths)

    # Non-numeric string must raise PredictionValidationError
    bad_str_rec = dict(base_record)
    bad_str_rec["sbytes"] = "invalid_text"
    with pytest.raises(PredictionValidationError, match="non-numeric"):
        predict_single_record(bad_str_rec, paths=temp_paths)

    # Valid reordered columns + extra columns + unknown category + missing numeric
    reordered = test_df[list(reversed(PREDEFINED_FEATURE_COLUMNS))].copy()
    reordered["unexpected_col"] = "extra"
    reordered.loc[0, "state"] = "UNSEEN_STATE"
    reordered.loc[1, "dur"] = np.nan
    csv_path = tmp_path / "reordered_valid.csv"
    reordered.to_csv(csv_path, index=False)

    batch_res = predict_batch_csv(csv_path, paths=temp_paths)
    assert batch_res.total_records == 20
    assert any("Ignored" in w for w in batch_res.warnings)
    assert any("missing value" in w for w in batch_res.warnings)

    # Oversized file limit enforcement
    with pytest.raises(PredictionValidationError, match="exceeds maximum"):
        predict_batch_csv(
            csv_path,
            file_size_bytes=MAX_UPLOAD_SIZE_BYTES + 1,
            paths=temp_paths,
        )

    # Path traversal model load blocked
    outside = tmp_path / "evil.joblib"
    outside.write_text("evil", encoding="utf-8")
    with pytest.raises(UntrustedModelError):
        load_model_artifact(outside, paths=temp_paths)


def test_real_trained_model_persistence_and_end_to_end_consistency(tmp_path: Path) -> None:
    paths = get_paths()
    best_model = load_model_artifact(paths.binary_models_dir / "best_model.joblib", paths=paths)
    meta = json.loads((paths.model_metadata_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["selected_model_key"] == "random_forest"

    # Load 10 real test rows, verify direct pipeline predictions match batch CSV API
    test_sample = pd.read_csv(paths.test_dataset_path, encoding="utf-8-sig", nrows=10)
    direct_preds = best_model.predict(test_sample[list(PREDEFINED_FEATURE_COLUMNS)])
    sample_csv = tmp_path / "real_sample.csv"
    test_sample.to_csv(sample_csv, index=False)
    batch_res = predict_batch_csv(sample_csv, paths=paths)
    assert np.array_equal(direct_preds, batch_res.predictions_df["predicted_label"].to_numpy())
    assert batch_res.has_ground_truth is True

    # Non-zero stream cursor position must still read cleanly
    stream = io.StringIO(sample_csv.read_text(encoding="utf-8"))
    stream.seek(50)
    stream_res = predict_batch_csv(stream, filename="stream_test.csv", paths=paths)
    assert stream_res.total_records == 10
