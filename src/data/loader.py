"""Reusable, memory-efficient dataset loader and profiler for UNSW-NB15."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import (
    BINARY_TARGET_COLUMN,
    ID_COLUMN,
    MULTICLASS_TARGET_COLUMN,
    ProjectPaths,
    get_paths,
)

# Normalized column names for the 49-column raw CSV parts (UNSW-NB15_1..4.csv)
RAW_49_COLUMNS: tuple[str, ...] = (
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "sjit",
    "djit",
    "stime",
    "ltime",
    "sintpkt",
    "dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "label",
)

# Mapping from 49-col feature descriptions to 45-col predefined set column names
_RAW_TO_PREDEFINED_ALIAS: dict[str, str] = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
    "sintpkt": "sinpkt",
    "dintpkt": "dinpkt",
}

_EXTRA_PREDEFINED_METADATA: dict[str, dict[str, str]] = {
    "id": {
        "type": "integer",
        "description": "Sequential record identifier in predefined train/test sets (excluded from ML features).",
    },
    "rate": {
        "type": "float",
        "description": "Packets per second flow rate derived in the predefined UNSW-NB15 train/test partitions.",
    },
}


class DatasetLoadError(RuntimeError):
    """Raised when a dataset file cannot be found, decoded, or parsed."""


def _read_csv_with_encoding_fallback(
    file_path: Path,
    *,
    preferred_encodings: tuple[str, ...] = ("utf-8-sig", "latin1"),
    **read_kwargs: Any,
) -> Any:
    """Attempt pd.read_csv across candidate encodings safely."""
    last_error: UnicodeDecodeError | None = None
    for encoding in preferred_encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, **read_kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DatasetLoadError(
        f"Unable to decode CSV file '{file_path.name}' using encodings {preferred_encodings}."
    ) from last_error


def _downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Safely downcast integer and float columns in-place on a copy to reduce RAM usage."""
    result = df.copy()
    for col in result.columns:
        series = result[col]
        if pd.api.types.is_integer_dtype(series):
            result[col] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            result[col] = pd.to_numeric(series, downcast="float")
    return result


def load_feature_descriptions(
    file_path: Path | None = None,
    *,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Load and normalize NUSW-NB15_features.csv metadata."""
    resolved_paths = paths or get_paths()
    target_path = file_path or resolved_paths.features_description_path
    if target_path is None or not target_path.is_file():
        raise DatasetLoadError(
            "Feature description file 'NUSW-NB15_features.csv' was not found."
        )

    try:
        df = _read_csv_with_encoding_fallback(
            target_path,
            preferred_encodings=("latin1", "utf-8-sig"),
        )
    except Exception as exc:
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(
            f"Failed to parse feature description file '{target_path.name}': {exc}"
        ) from exc

    df.columns = [str(col).strip() for col in df.columns]
    if "Name" not in df.columns or "Type" not in df.columns or "Description" not in df.columns:
        raise DatasetLoadError(
            f"Unexpected columns in '{target_path.name}': {list(df.columns)}"
        )

    df["Name"] = df["Name"].astype(str).str.strip()
    df["Type"] = df["Type"].astype(str).str.strip().str.lower()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["normalized_name"] = (
        df["Name"]
        .str.replace("ct_src_ ltm", "ct_src_ltm", regex=False)
        .str.lower()
    )
    df["predefined_name"] = df["normalized_name"].replace(_RAW_TO_PREDEFINED_ALIAS)
    return df


def get_feature_metadata_map(
    file_path: Path | None = None,
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, dict[str, str]]:
    """Build a lookup map from column names (45-col and 49-col) to type & description."""
    features_df = load_feature_descriptions(file_path=file_path, paths=paths)
    mapping: dict[str, dict[str, str]] = {}

    for row in features_df.itertuples(index=False):
        entry = {"type": str(row.Type), "description": str(row.Description)}
        mapping[str(row.normalized_name)] = entry
        mapping[str(row.predefined_name)] = entry

    for col_name, info in _EXTRA_PREDEFINED_METADATA.items():
        mapping.setdefault(col_name, dict(info))
    return mapping


def load_csv_dataset(
    file_path: Path,
    *,
    has_header: bool = True,
    column_names: Sequence[str] | None = None,
    nrows: int | None = None,
    downcast_numeric: bool = False,
) -> pd.DataFrame:
    """Load a CSV dataset with safe encoding handling and optional numeric downcasting."""
    if not file_path.is_file():
        raise DatasetLoadError(f"Dataset file not found: '{file_path.name}'.")

    read_kwargs: dict[str, Any] = {
        "low_memory": False,
        "nrows": nrows,
    }
    if has_header:
        read_kwargs["header"] = 0
        if column_names is not None:
            read_kwargs["names"] = list(column_names)
    else:
        read_kwargs["header"] = None
        if column_names is not None:
            read_kwargs["names"] = list(column_names)

    try:
        df = _read_csv_with_encoding_fallback(file_path, **read_kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError(f"CSV file '{file_path.name}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError(
            f"Malformed CSV format in '{file_path.name}': {exc}"
        ) from exc
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(
            f"Unexpected error reading '{file_path.name}': {exc}"
        ) from exc

    if downcast_numeric:
        df = _downcast_numeric_columns(df)
    return df


def iter_csv_chunks(
    file_path: Path,
    *,
    chunksize: int = 100_000,
    has_header: bool = True,
    column_names: Sequence[str] | None = None,
    downcast_numeric: bool = False,
) -> Iterator[pd.DataFrame]:
    """Stream a large CSV file in memory-bounded chunks."""
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    if not file_path.is_file():
        raise DatasetLoadError(f"Dataset file not found: '{file_path.name}'.")

    read_kwargs: dict[str, Any] = {
        "chunksize": chunksize,
        "low_memory": False,
        "header": 0 if has_header else None,
    }
    if column_names is not None:
        read_kwargs["names"] = list(column_names)

    reader = _read_csv_with_encoding_fallback(file_path, **read_kwargs)
    for chunk in reader:
        yield _downcast_numeric_columns(chunk) if downcast_numeric else chunk


def _normalize_raw_part_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known whitespace/label inconsistencies in headerless 49-col raw splits."""
    cleaned = df.copy()
    if MULTICLASS_TARGET_COLUMN in cleaned.columns:
        cat_series = cleaned[MULTICLASS_TARGET_COLUMN]
        cat_series = cat_series.where(cat_series.notna(), "Normal")
        cat_series = cat_series.astype(str).str.strip()
        cat_series = cat_series.replace({"": "Normal", "nan": "Normal", "Backdoors": "Backdoor"})
        cleaned[MULTICLASS_TARGET_COLUMN] = cat_series

    for col in ("proto", "state", "service", "ct_ftp_cmd"):
        if col in cleaned.columns and (
            pd.api.types.is_string_dtype(cleaned[col])
            or pd.api.types.is_object_dtype(cleaned[col])
        ):
            cleaned[col] = cleaned[col].astype(str).str.strip()
    return cleaned


def load_raw_part(
    part_number: int,
    *,
    nrows: int | None = None,
    downcast_numeric: bool = False,
    normalize_labels: bool = True,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Load one of the four raw 49-column UNSW-NB15 CSV split files (1..4)."""
    if part_number not in (1, 2, 3, 4):
        raise ValueError("part_number must be between 1 and 4.")

    resolved_paths = paths or get_paths()
    filename = f"UNSW-NB15_{part_number}.csv"
    file_path = resolved_paths.raw_part_paths.get(filename)
    if file_path is None or not file_path.is_file():
        raise DatasetLoadError(f"Raw split file '{filename}' was not found.")

    df = load_csv_dataset(
        file_path,
        has_header=False,
        column_names=RAW_49_COLUMNS,
        nrows=nrows,
        downcast_numeric=downcast_numeric,
    )
    return _normalize_raw_part_chunk(df) if normalize_labels else df


def load_training_and_testing_sets(
    *,
    nrows_train: int | None = None,
    nrows_test: int | None = None,
    downcast_numeric: bool = False,
    paths: ProjectPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the predefined UNSW-NB15 training and testing CSV datasets."""
    resolved_paths = paths or get_paths()
    train_path = resolved_paths.train_dataset_path
    test_path = resolved_paths.test_dataset_path

    if train_path is None or not train_path.is_file():
        raise DatasetLoadError(
            "Predefined training dataset 'UNSW_NB15_training-set.csv' was not found."
        )
    if test_path is None or not test_path.is_file():
        raise DatasetLoadError(
            "Predefined testing dataset 'UNSW_NB15_testing-set.csv' was not found."
        )

    train_df = load_csv_dataset(
        train_path,
        has_header=True,
        nrows=nrows_train,
        downcast_numeric=downcast_numeric,
    )
    test_df = load_csv_dataset(
        test_path,
        has_header=True,
        nrows=nrows_test,
        downcast_numeric=downcast_numeric,
    )
    return train_df, test_df


def profile_dataset(
    df: pd.DataFrame,
    *,
    name: str = "dataset",
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive dataset metadata, nulls, duplicates, and target counts."""
    num_records = int(len(df))
    num_columns = int(df.shape[1])

    missing_series = df.isna().sum()
    missing_by_col = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
    total_missing = int(missing_series.sum())

    dash_placeholders: dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            dash_count = int((df[col].astype(str).str.strip() == "-").sum())
            if dash_count > 0:
                dash_placeholders[col] = dash_count

    exact_duplicates = int(df.duplicated().sum())
    if ID_COLUMN in df.columns:
        non_id_cols = [c for c in df.columns if c != ID_COLUMN]
        duplicates_excl_id = int(df.duplicated(subset=non_id_cols).sum())
    else:
        duplicates_excl_id = exact_duplicates

    if feature_columns is not None:
        feat_subset = [c for c in feature_columns if c in df.columns]
    else:
        excluded = {ID_COLUMN, MULTICLASS_TARGET_COLUMN, BINARY_TARGET_COLUMN}
        feat_subset = [c for c in df.columns if c not in excluded]
    feature_only_duplicates = int(df.duplicated(subset=feat_subset).sum()) if feat_subset else 0

    target_distribution: dict[str, Any] = {}
    if BINARY_TARGET_COLUMN in df.columns:
        label_counts = {
            str(k): int(v)
            for k, v in df[BINARY_TARGET_COLUMN].value_counts(dropna=False).items()
        }
        label_pct = {
            str(k): round(float(v) / num_records * 100.0, 2) if num_records else 0.0
            for k, v in label_counts.items()
        }
        target_distribution["binary_counts"] = label_counts
        target_distribution["binary_percentages"] = label_pct

    if MULTICLASS_TARGET_COLUMN in df.columns:
        cat_counts = {
            str(k): int(v)
            for k, v in df[MULTICLASS_TARGET_COLUMN].value_counts(dropna=False).items()
        }
        cat_pct = {
            str(k): round(float(v) / num_records * 100.0, 2) if num_records else 0.0
            for k, v in cat_counts.items()
        }
        target_distribution["attack_cat_counts"] = cat_counts
        target_distribution["attack_cat_percentages"] = cat_pct

    memory_bytes = int(df.memory_usage(deep=True).sum())

    return {
        "name": name,
        "num_records": num_records,
        "num_columns": num_columns,
        "memory_usage_mb": round(memory_bytes / (1024 * 1024), 2),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            "total_missing_cells": total_missing,
            "columns_with_missing": missing_by_col,
        },
        "dash_placeholders": dash_placeholders,
        "duplicate_statistics": {
            "exact_duplicates": exact_duplicates,
            "duplicates_excluding_id": duplicates_excl_id,
            "feature_only_duplicates": feature_only_duplicates,
        },
        "target_distribution": target_distribution,
    }
