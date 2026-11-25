"""Cached data and metadata access helpers for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data.loader import (
    DatasetLoadError,
    get_feature_metadata_map,
    load_csv_dataset,
)
from src.utils.config import get_paths


@st.cache_data(show_spinner=False)
def load_json_artifact(relative_path_str: str) -> dict[str, Any] | None:
    """Load a JSON metadata/report file relative to project root if present."""
    paths = get_paths()
    target = (paths.root / relative_path_str).resolve()
    try:
        target.relative_to(paths.root.resolve())
    except ValueError:
        return None
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_sample_split(split: str = "train", nrows: int = 5000) -> pd.DataFrame | None:
    """Load a bounded sample of the training or testing CSV for exploration."""
    paths = get_paths()
    target = paths.train_dataset_path if split == "train" else paths.test_dataset_path
    if target is None or not target.is_file():
        return None
    try:
        return load_csv_dataset(target, has_header=True, nrows=nrows, downcast_numeric=True)
    except DatasetLoadError:
        return None


@st.cache_data(show_spinner=False)
def load_feature_descriptions_map() -> dict[str, dict[str, str]]:
    """Return column -> {type, description} mapping from NUSW-NB15_features.csv."""
    try:
        return get_feature_metadata_map()
    except DatasetLoadError:
        return {}


@st.cache_data(show_spinner=False)
def get_categorical_value_choices() -> dict[str, list[str]]:
    """Discover real categorical options from the predefined training dataset."""
    paths = get_paths()
    train_path = paths.train_dataset_path
    defaults = {
        "proto": ["tcp", "udp", "unas", "arp", "ospf", "icmp"],
        "service": ["-", "dns", "http", "smtp", "ftp-data", "ftp", "ssh", "pop3"],
        "state": ["FIN", "INT", "CON", "REQ", "RST"],
    }
    if train_path is None or not train_path.is_file():
        return defaults
    try:
        df = pd.read_csv(
            train_path,
            usecols=["proto", "service", "state"],
            encoding="utf-8-sig",
        )
        return {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in ("proto", "service", "state")
        }
    except Exception:
        return defaults


def get_artifact_status() -> dict[str, bool]:
    """Check presence of datasets, reports, binary models, and multiclass models."""
    paths = get_paths()
    return {
        "train_dataset": paths.train_dataset_path is not None,
        "test_dataset": paths.test_dataset_path is not None,
        "features_file": paths.features_description_path is not None,
        "inspection_report": (paths.reports_dir / "dataset_inspection_report.json").is_file(),
        "eda_summary": (paths.reports_dir / "eda_summary.json").is_file(),
        "preprocessing_config": (paths.model_metadata_dir / "preprocessing_config.json").is_file(),
        "training_summary": (paths.model_metadata_dir / "binary_training_summary.json").is_file(),
        "cv_results": (paths.model_metadata_dir / "cv_results.json").is_file(),
        "tuning_results": (paths.model_metadata_dir / "tuning_results.json").is_file(),
        "evaluation_results": (paths.model_metadata_dir / "binary_evaluation_results.json").is_file(),
        "model_metadata": (paths.model_metadata_dir / "model_metadata.json").is_file(),
        "best_binary_model": (paths.binary_models_dir / "best_model.joblib").is_file(),
        "multiclass_model": (paths.multiclass_models_dir / "best_model.joblib").is_file(),
        "multiclass_metadata": (paths.model_metadata_dir / "multiclass_metadata.json").is_file(),
    }


def list_saved_figures() -> list[tuple[str, Path]]:
    """Return sorted list of (label, Path) for generated PNG visualizations."""
    paths = get_paths()
    if not paths.reports_figures_dir.is_dir():
        return []
    files = sorted(paths.reports_figures_dir.glob("*.png"))
    return [(f.stem.replace("_", " ").title(), f) for f in files]
