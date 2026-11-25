# NETGUARD AI — Codebase Cleanup & Optimization Inspection Report

- **Inspection Date:** 2026-11-26
- **Scope:** Full static AST, dependency, duplicate-code, and filesystem audit of `D:\Project\NETGUARD-AI`
- **Status:** **Awaiting User Approval Before Any Deletion or Refactoring**

---

## 1. Protected Assets (Will NOT Be Deleted or Modified)
- **Original UNSW-NB15 Datasets:** `Training and Testing Sets/`, `UNSW-NB15_1.csv`–`4.csv`, `NUSW-NB15_features.csv`, `NUSW-NB15_GT.csv`, `UNSW-NB15_LIST_EVENTS.csv`, `ReadMe.pdf`, `The UNSW-NB15 description.pdf`
- **Trained ML Models & Metadata:** `models/binary/*.joblib`, `models/multiclass/*.joblib`, `models/metadata/*.json`
- **Active Source & UI Modules:** `src/`, `app/`, `main.py`, `pyproject.toml`, `.streamlit/config.toml`, `.gitignore`
- **Tests, Notebooks & Documentation:** `tests/`, `notebooks/01`–`06`, `README.md`, `reports/MCA_PBL_PROJECT_REPORT.md`, `reports/final_audit/`, `reports/figures/*.png`, `reports/*.json`
- **Git-Ignored Directory Placeholders:** `dataset/raw/.gitkeep`, `dataset/processed/.gitkeep`, `database/.gitkeep`, `models/binary/.gitkeep`, `models/multiclass/.gitkeep`, `reports/figures/.gitkeep` (required by `.gitignore` to preserve directory structure on fresh clone)

---

## 2. Proposed File & Directory Deletions (Requires Approval)

| Item | Type | Reason Considered Unnecessary |
|---|---|---|
| `app/components/.gitkeep` | Obsolete placeholder | Directory already tracks `__init__.py`, `charts.py`, `data_access.py`, `theme.py` |
| `models/metadata/.gitkeep` | Obsolete placeholder | Directory already tracks 7 persistent JSON metadata files |
| `notebooks/.gitkeep` | Obsolete placeholder | Directory already tracks 6 `.ipynb` files (`01`–`06`) |
| `app/pages/.gitkeep` & `app/pages/` | Unused empty folder | Navigation routes cleanly via `st.navigation` to `app/views/` |
| `app/assets/.gitkeep` & `app/assets/` | Unused empty folder | No static assets stored or referenced |
| `__pycache__/` (14 dirs) & `.pytest_cache/` | Ephemeral cache artifacts | Auto-generated Python bytecode and Pytest cache (not tracked in Git) |

*(Note: `database/netguard_history.db` is empty `0` rows; retained unless user explicitly requests resetting DB file.)*

---

## 3. Proposed Unused Imports & Dead Arguments Cleanup

### Unused Imports (11 verified via AST)
1. `app/views/history_reports_system.py:5` — `from pathlib import Path`
2. `app/views/prediction_views.py:5` — `import pandas as pd`
3. `src/data/loader.py:9` — `import numpy as np`
4. `src/prediction/predict.py:17` — `PREDEFINED_NUMERICAL_COLUMNS`
5. `src/training/cross_validation.py:8` — `from pathlib import Path`
6. `src/training/multiclass.py:8` — `from pathlib import Path`
7. `src/training/selector.py:7` — `from pathlib import Path`
8. `src/training/train.py:9` — `from pathlib import Path`
9. `src/training/tuning.py:8` — `from pathlib import Path`
10. `tests/test_e2e_user_journey.py:9` — `clear_history`
11. `tests/test_models_training_and_evaluation.py:5` — `import json`

### Inline Imports to Move to Module Top-Level (PEP 8)
- `app/views/history_reports_system.py` (lines 134 & 224): repeated `import pandas as pd` inside `render_reports_exports()` and `render_settings()`
- `app/views/prediction_views.py` (line 86): `from src.utils.config import get_paths` inside `render_single_prediction()`

### Unused Test Function Parameters
- `tests/test_data_loader_and_validator.py:176`: unused `tmp_path` argument in `test_cli_inspect_data_saves_report_and_passes`
- `tests/test_e2e_user_journey.py:13`: unused `tmp_path, monkeypatch` arguments in `test_interactive_e2e_single_prediction_history_and_settings`

---

## 4. Proposed Duplicate Code Refactoring
1. **Unify 3 Duplicate Stratified Sampling Implementations:**
   - Currently duplicated across:
     - `src/data/eda.py` (`_stratified_sample`)
     - `src/training/cross_validation.py` (`_stratified_subsample`, also privately imported by `src/training/tuning.py`)
     - `src/training/train.py` (inline `groupby("_target").sample(...)` inside `evaluate_class_weight_effect`)
   - **Optimization:** Extract unified public helpers `stratified_sample_df(...)` and `stratified_sample_xy(...)` in `src/data/loader.py` (exported in `src/data/__init__.py`) and reuse across `eda.py`, `train.py`, `cross_validation.py`, and `tuning.py`.
2. **Synchronize `README.md` Directory Tree:**
   - Update `README.md` Section 7 tree if `app/pages/` and `app/assets/` are removed.

---

## 5. Dependency Audit (`requirements.txt`)
- All 9 entries in `requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `joblib`, `pytest`) are actively imported across `src/`, `app/`, and `tests/`. **0 unused dependencies to remove from `requirements.txt`.**
- `.venv` also contains `ipykernel==7.4.0` (used for Jupyter notebook kernel support, intentionally omitted from runtime `requirements.txt`).
