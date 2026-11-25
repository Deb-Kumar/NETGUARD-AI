"""Phase 2 tests: verify project structure, Python 3.12 environment, and config."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from main import collect_dependency_versions, main as cli_main
from src.utils.config import (
    BINARY_TARGET_COLUMN,
    DEFAULT_N_JOBS,
    MULTICLASS_TARGET_COLUMN,
    NON_FEATURE_COLUMNS,
    RANDOM_STATE,
    ensure_directories,
    get_environment_summary,
    get_paths,
)


def test_python_version_is_3_12() -> None:
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_required_dependencies_importable() -> None:
    modules = (
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
        "joblib",
        "pytest",
        "sqlite3",
    )
    for mod_name in modules:
        imported = importlib.import_module(mod_name)
        assert imported is not None

    versions = collect_dependency_versions()
    assert all(ver is not None for ver in versions.values())


def test_project_directory_structure_exists() -> None:
    paths = get_paths()
    expected_dirs = (
        paths.dataset_dir,
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.src_dir,
        paths.app_dir,
        paths.models_dir,
        paths.binary_models_dir,
        paths.multiclass_models_dir,
        paths.model_metadata_dir,
        paths.database_dir,
        paths.reports_dir,
        paths.reports_figures_dir,
        paths.notebooks_dir,
        paths.tests_dir,
    )
    for directory in expected_dirs:
        assert directory.is_dir(), f"Missing directory: {directory}"

    assert (paths.root / ".gitignore").is_file()
    assert (paths.root / "requirements.txt").is_file()
    assert (paths.root / "README.md").is_file()
    assert (paths.root / "main.py").is_file()


def test_dataset_paths_resolve_without_copying() -> None:
    paths = get_paths()
    assert paths.train_dataset_path is not None
    assert paths.train_dataset_path.is_file()
    assert paths.test_dataset_path is not None
    assert paths.test_dataset_path.is_file()
    assert paths.features_description_path is not None
    assert paths.features_description_path.is_file()
    assert all(p is not None and p.is_file() for p in paths.raw_part_paths.values())


def test_config_constants_prevent_leakage_and_bound_workers() -> None:
    assert RANDOM_STATE == 42
    assert 1 <= DEFAULT_N_JOBS <= 4
    assert "id" in NON_FEATURE_COLUMNS
    assert BINARY_TARGET_COLUMN in NON_FEATURE_COLUMNS
    assert MULTICLASS_TARGET_COLUMN in NON_FEATURE_COLUMNS


def test_ensure_directories_is_idempotent_in_temp_root(tmp_path: Path) -> None:
    temp_paths = get_paths(tmp_path)
    ensure_directories(temp_paths)
    assert temp_paths.binary_models_dir.is_dir()
    assert temp_paths.reports_figures_dir.is_dir()
    assert temp_paths.train_dataset_path is None


def test_cli_check_env_succeeds(capsys) -> None:
    exit_code = cli_main(["check-env"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["train_dataset_found"] is True
    assert payload["test_dataset_found"] is True
    assert payload["features_file_found"] is True
    summary = get_environment_summary()
    assert summary["project_name"] == "NETGUARD AI"
