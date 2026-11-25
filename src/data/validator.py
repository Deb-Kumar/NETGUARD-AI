"""Dataset schema validation and target-leakage safeguards for NETGUARD AI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils.config import (
    BINARY_TARGET_COLUMN,
    ID_COLUMN,
    MULTICLASS_TARGET_COLUMN,
    NON_FEATURE_COLUMNS,
)

# Exact 45-column schema verified on UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv
PREDEFINED_COLUMNS: tuple[str, ...] = (
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
)

PREDEFINED_CATEGORICAL_COLUMNS: tuple[str, ...] = ("proto", "service", "state")

PREDEFINED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for col in PREDEFINED_COLUMNS if col not in NON_FEATURE_COLUMNS
)

PREDEFINED_NUMERICAL_COLUMNS: tuple[str, ...] = tuple(
    col
    for col in PREDEFINED_FEATURE_COLUMNS
    if col not in PREDEFINED_CATEGORICAL_COLUMNS
)


class DataLeakageError(ValueError):
    """Raised when identifier or target-derived columns appear in model features."""


@dataclass
class ValidationResult:
    """Structured validation outcome containing errors, warnings, and metrics."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation results into JSON-friendly dictionary form."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


def validate_no_target_leakage(
    feature_columns: Sequence[str],
    *,
    task: Literal["binary", "multiclass"] = "binary",
) -> None:
    """Ensure identifier and target columns are never passed as input features."""
    normalized = {str(col).strip().lower() for col in feature_columns}
    forbidden = {ID_COLUMN.lower(), BINARY_TARGET_COLUMN.lower(), MULTICLASS_TARGET_COLUMN.lower()}
    leaked = sorted(normalized & forbidden)
    if leaked:
        raise DataLeakageError(
            f"Target/identifier leakage detected for '{task}' task: "
            f"forbidden columns {leaked} found in feature set."
        )


def extract_features_and_target(
    df: pd.DataFrame,
    *,
    task: Literal["binary", "multiclass"] = "binary",
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract leakage-free feature matrix X and target vector y from a DataFrame."""
    target_col = (
        BINARY_TARGET_COLUMN if task == "binary" else MULTICLASS_TARGET_COLUMN
    )
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    selected_features = (
        list(feature_columns)
        if feature_columns is not None
        else [col for col in PREDEFINED_FEATURE_COLUMNS if col in df.columns]
    )
    if not selected_features:
        raise ValueError("No valid feature columns could be selected.")

    validate_no_target_leakage(selected_features, task=task)

    missing_feats = [col for col in selected_features if col not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing expected feature columns: {missing_feats}")

    X = df[selected_features].copy()
    y = df[target_col].copy()
    return X, y


def validate_predefined_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    require_targets: bool = True,
) -> ValidationResult:
    """Validate schema, dtypes, numeric finiteness, and target consistency."""
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append(f"{dataset_name}: Dataset is empty (0 rows).")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    expected_cols = (
        list(PREDEFINED_COLUMNS)
        if require_targets
        else list(PREDEFINED_FEATURE_COLUMNS)
    )
    actual_cols = list(df.columns)
    missing_cols = [c for c in expected_cols if c not in actual_cols]
    extra_cols = [c for c in actual_cols if c not in PREDEFINED_COLUMNS]

    if missing_cols:
        errors.append(f"{dataset_name}: Missing required columns: {missing_cols}")
    if extra_cols:
        warnings.append(f"{dataset_name}: Unexpected extra columns present: {extra_cols}")

    if require_targets and actual_cols != expected_cols and not missing_cols:
        warnings.append(
            f"{dataset_name}: Column order differs from canonical UNSW-NB15 schema."
        )

    # Validate numerical feature columns
    non_numeric_cols: list[str] = []
    inf_cols: list[str] = []
    for col in PREDEFINED_NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
        else:
            values = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(values).any():
                inf_cols.append(col)

    if non_numeric_cols:
        errors.append(
            f"{dataset_name}: Expected numeric dtype for columns: {non_numeric_cols}"
        )
    if inf_cols:
        errors.append(
            f"{dataset_name}: Infinite (+/-inf) values detected in columns: {inf_cols}"
        )

    # Check ID uniqueness if present
    if ID_COLUMN in df.columns:
        if df[ID_COLUMN].isna().any():
            errors.append(f"{dataset_name}: Null values detected in '{ID_COLUMN}'.")
        elif df[ID_COLUMN].duplicated().any():
            warnings.append(
                f"{dataset_name}: Duplicate '{ID_COLUMN}' values found."
            )

    # Validate binary and multiclass targets when required or present
    if BINARY_TARGET_COLUMN in df.columns:
        label_series = df[BINARY_TARGET_COLUMN]
        if label_series.isna().any():
            errors.append(f"{dataset_name}: Null values found in '{BINARY_TARGET_COLUMN}'.")
        else:
            unique_labels = set(label_series.unique().tolist())
            if not unique_labels.issubset({0, 1}):
                errors.append(
                    f"{dataset_name}: Invalid binary '{BINARY_TARGET_COLUMN}' values "
                    f"{sorted(unique_labels)}; expected subset of {{0, 1}}."
                )
    elif require_targets:
        errors.append(f"{dataset_name}: Missing target column '{BINARY_TARGET_COLUMN}'.")

    if MULTICLASS_TARGET_COLUMN in df.columns:
        cat_series = df[MULTICLASS_TARGET_COLUMN]
        if cat_series.isna().any():
            errors.append(
                f"{dataset_name}: Null values found in '{MULTICLASS_TARGET_COLUMN}'."
            )
        elif BINARY_TARGET_COLUMN in df.columns and not df[BINARY_TARGET_COLUMN].isna().any():
            is_zero = df[BINARY_TARGET_COLUMN] == 0
            is_normal = cat_series.astype(str).str.strip() == "Normal"
            mismatch_count = int((is_zero != is_normal).sum())
            if mismatch_count > 0:
                errors.append(
                    f"{dataset_name}: {mismatch_count} rows violate "
                    "'label == 0 <-> attack_cat == Normal' consistency."
                )
    elif require_targets:
        errors.append(
            f"{dataset_name}: Missing multiclass target column '{MULTICLASS_TARGET_COLUMN}'."
        )

    # Check non-id duplicate rows (expected in UNSW-NB15 after dropping IPs/ports/time)
    non_id_cols = [c for c in df.columns if c != ID_COLUMN]
    dup_excl_id = int(df.duplicated(subset=non_id_cols).sum()) if non_id_cols else 0
    if dup_excl_id > 0:
        warnings.append(
            f"{dataset_name}: Contains {dup_excl_id} duplicate rows when excluding '{ID_COLUMN}'."
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary={
            "dataset_name": dataset_name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicates_excluding_id": dup_excl_id,
        },
    )


def validate_train_test_compatibility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> ValidationResult:
    """Validate that predefined train and test splits are compatible and leakage-checked."""
    train_res = validate_predefined_dataset(train_df, dataset_name="train")
    test_res = validate_predefined_dataset(test_df, dataset_name="test")

    errors = list(train_res.errors) + list(test_res.errors)
    warnings = list(train_res.warnings) + list(test_res.warnings)

    if list(train_df.columns) != list(test_df.columns):
        errors.append(
            "Train and test datasets do not share identical column names/ordering."
        )

    unseen_categories: dict[str, list[str]] = {}
    for cat_col in PREDEFINED_CATEGORICAL_COLUMNS:
        if cat_col in train_df.columns and cat_col in test_df.columns:
            train_vals = set(train_df[cat_col].astype(str).unique())
            test_vals = set(test_df[cat_col].astype(str).unique())
            unseen = sorted(test_vals - train_vals)
            if unseen:
                unseen_categories[cat_col] = unseen
                warnings.append(
                    f"Test column '{cat_col}' has unseen categories not in train: {unseen} "
                    "(requires handle_unknown='ignore')."
                )

    # Check identical feature-vector overlap between train and test for documentation/EDA
    shared_features = [c for c in PREDEFINED_FEATURE_COLUMNS if c in train_df.columns and c in test_df.columns]
    overlap_count = 0
    if shared_features and not errors:
        train_hashes = set(
            pd.util.hash_pandas_object(train_df[shared_features], index=False).tolist()
        )
        test_hashes = pd.util.hash_pandas_object(test_df[shared_features], index=False)
        overlap_count = int(test_hashes.isin(train_hashes).sum())
        if overlap_count > 0:
            warnings.append(
                f"{overlap_count} test rows share identical 42-feature signatures with training rows "
                "(known UNSW-NB15 artifact after removing IP/port/timestamp fields)."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary={
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "unseen_test_categories": unseen_categories,
            "feature_signature_overlap_rows": overlap_count,
        },
    )
