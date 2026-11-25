"""Central configuration and path resolution for NETGUARD AI."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME: str = "NETGUARD AI"
PROJECT_TITLE: str = (
    "AI-Powered Network Intrusion Detection and Threat Classification System "
    "Using Machine Learning"
)
APP_VERSION: str = "0.2.0"


def load_local_dotenv(env_file: Path | None = None) -> None:
    """Load key=value overrides from .env via os.environ.setdefault if present."""
    target = env_file or (Path(__file__).resolve().parents[2] / ".env")
    if not target.is_file():
        return
    for raw_line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val:
            os.environ.setdefault(key, val)


load_local_dotenv()


def _int_env(var_name: str, default: int, *, min_val: int = 1, max_val: int = 1024) -> int:
    """Safely parse a bounded integer environment override."""
    raw = os.getenv(var_name)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(min_val, min(max_val, parsed))


# Reproducibility and resource defaults
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
# Bounded default worker count to prevent memory exhaustion on Windows systems.
DEFAULT_N_JOBS: int = _int_env("NETGUARD_N_JOBS", 2, min_val=1, max_val=8)
MAX_UPLOAD_SIZE_MB: int = _int_env("NETGUARD_MAX_UPLOAD_MB", 50, min_val=1, max_val=200)
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Dataset schema constants (verified against UNSW-NB15 training/testing CSVs)
ID_COLUMN: str = "id"
BINARY_TARGET_COLUMN: str = "label"
MULTICLASS_TARGET_COLUMN: str = "attack_cat"
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    MULTICLASS_TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
)
BINARY_CLASS_NAMES: dict[int, str] = {
    0: "Normal Traffic",
    1: "Malicious Traffic",
}

# Exact dataset filenames observed in the UNSW-NB15 distribution
TRAIN_SET_FILENAME: str = "UNSW_NB15_training-set.csv"
TEST_SET_FILENAME: str = "UNSW_NB15_testing-set.csv"
FEATURES_FILENAME: str = "NUSW-NB15_features.csv"
GROUND_TRUTH_FILENAME: str = "NUSW-NB15_GT.csv"
LIST_EVENTS_FILENAME: str = "UNSW-NB15_LIST_EVENTS.csv"
RAW_PART_FILENAMES: tuple[str, ...] = (
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
)


def get_project_root() -> Path:
    """Return the resolved root directory of the NETGUARD AI project."""
    env_root = os.getenv("NETGUARD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths for project modules, datasets, and artifacts."""

    root: Path
    dataset_dir: Path
    raw_data_dir: Path
    raw_csv_dir: Path
    raw_docs_dir: Path
    processed_data_dir: Path
    training_testing_dir: Path
    src_dir: Path
    app_dir: Path
    models_dir: Path
    binary_models_dir: Path
    multiclass_models_dir: Path
    model_metadata_dir: Path
    database_dir: Path
    db_path: Path
    reports_dir: Path
    reports_graphs_dir: Path
    reports_experiments_dir: Path
    logs_dir: Path
    notebooks_dir: Path
    tests_dir: Path

    def resolve_dataset_file(self, filename: str) -> Path | None:
        """Locate a dataset file without modifying or copying original files.

        Searches organized ``dataset/raw/csv/`` directories first, followed by
        legacy root locations for backward compatibility.
        """
        candidates = (
            self.raw_csv_dir / "Training and Testing Sets" / filename,
            self.raw_csv_dir / filename,
            self.raw_docs_dir / filename,
            self.raw_data_dir / filename,
            self.training_testing_dir / filename,
            self.root / filename,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @property
    def train_dataset_path(self) -> Path | None:
        """Return the resolved path to the predefined UNSW-NB15 training CSV."""
        return self.resolve_dataset_file(TRAIN_SET_FILENAME)

    @property
    def test_dataset_path(self) -> Path | None:
        """Return the resolved path to the predefined UNSW-NB15 testing CSV."""
        return self.resolve_dataset_file(TEST_SET_FILENAME)

    @property
    def features_description_path(self) -> Path | None:
        """Return the resolved path to NUSW-NB15_features.csv."""
        return self.resolve_dataset_file(FEATURES_FILENAME)

    @property
    def list_events_path(self) -> Path | None:
        """Return the resolved path to UNSW-NB15_LIST_EVENTS.csv."""
        return self.resolve_dataset_file(LIST_EVENTS_FILENAME)

    @property
    def ground_truth_path(self) -> Path | None:
        """Return the resolved path to NUSW-NB15_GT.csv."""
        return self.resolve_dataset_file(GROUND_TRUTH_FILENAME)

    @property
    def raw_part_paths(self) -> dict[str, Path | None]:
        """Return resolved paths for the four raw UNSW-NB15 CSV split files."""
        return {name: self.resolve_dataset_file(name) for name in RAW_PART_FILENAMES}


def get_paths(project_root: Path | None = None) -> ProjectPaths:
    """Build a ProjectPaths instance rooted at ``project_root`` (or default root)."""
    root = project_root.resolve() if project_root is not None else get_project_root()
    dataset_dir = root / "dataset"
    raw_data_dir = dataset_dir / "raw"
    raw_csv_dir = raw_data_dir / "csv"
    models_dir = root / "models"
    database_dir = root / "database"
    reports_dir = root / "reports"
    return ProjectPaths(
        root=root,
        dataset_dir=dataset_dir,
        raw_data_dir=raw_data_dir,
        raw_csv_dir=raw_csv_dir,
        raw_docs_dir=raw_data_dir / "documentation",
        processed_data_dir=dataset_dir / "processed",
        training_testing_dir=raw_csv_dir / "Training and Testing Sets",
        src_dir=root / "src",
        app_dir=root / "app",
        models_dir=models_dir,
        binary_models_dir=models_dir / "binary",
        multiclass_models_dir=models_dir / "multiclass",
        model_metadata_dir=models_dir / "metadata",
        database_dir=database_dir,
        db_path=database_dir / "netguard_history.db",
        reports_dir=reports_dir,
        reports_graphs_dir=reports_dir / "graphs",
        reports_experiments_dir=reports_dir / "experiments",
        logs_dir=root / "logs",
        notebooks_dir=root / "notebooks",
        tests_dir=root / "tests",
    )


def ensure_directories(paths: ProjectPaths | None = None) -> ProjectPaths:
    """Ensure required output/working directories exist without touching raw files."""
    resolved = paths or get_paths()
    required_dirs = (
        resolved.raw_data_dir,
        resolved.raw_csv_dir,
        resolved.raw_docs_dir,
        resolved.processed_data_dir,
        resolved.binary_models_dir,
        resolved.multiclass_models_dir,
        resolved.model_metadata_dir,
        resolved.database_dir,
        resolved.reports_dir,
        resolved.reports_graphs_dir,
        resolved.reports_experiments_dir,
        resolved.logs_dir,
        resolved.notebooks_dir,
        resolved.tests_dir,
    )
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return resolved


def get_environment_summary(paths: ProjectPaths | None = None) -> dict[str, object]:
    """Collect runtime and dataset availability metadata for diagnostics."""
    resolved = paths or get_paths()
    return {
        "project_name": PROJECT_NAME,
        "version": APP_VERSION,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(resolved.root),
        "train_dataset_found": resolved.train_dataset_path is not None,
        "test_dataset_found": resolved.test_dataset_path is not None,
        "features_file_found": resolved.features_description_path is not None,
        "raw_parts_found": {
            name: path is not None for name, path in resolved.raw_part_paths.items()
        },
        "default_n_jobs": DEFAULT_N_JOBS,
        "random_state": RANDOM_STATE,
    }
