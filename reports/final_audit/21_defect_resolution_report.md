# 21 — Defect Resolution Report

## Defects Identified, Fixed & Verified

| ID | Severity | Area | Root Cause | Resolution Applied | Verification |
|---|---|---|---|---|---|
| **DEF-01** | **HIGH / MEDIUM** | Database (`src/database/history_db.py`) | `with sqlite3.connect(...) as conn:` only commits/rolls back on exit without closing the connection, causing Windows file handle leaks (`WinError 32`). | Wrapped all DB connections in `contextlib.closing(_connect(db_path))`. | `test_sqlite_connections_close_cleanly_on_windows_and_block_sqli` **PASSED** |
| **DEF-02** | **MEDIUM** | Security (`src/database/history_db.py`, `app/views/prediction_views.py`) | Exported CSVs in `export_history_csv()` and batch downloads did not neutralize spreadsheet formula prefixes (`=`, `+`, `@`, `-`). | Added `sanitize_csv_cell` & `sanitize_dataframe_for_csv` while preserving valid negative floats. | `test_csv_formula_sanitization_preserves_negative_numbers` **PASSED** |
| **DEF-03** | **MEDIUM** | Input Validation (`src/prediction/predict.py`) | `validate_and_align_features()` allowed `< 0` physical metrics and silently converted invalid text strings to `NaN`; `predict_batch_csv()` overwrote explicit `file_size_bytes` when given a `Path`. | Added strict rejection of negative numerical values and non-numeric strings (`PredictionValidationError`) and respected explicit `file_size_bytes`. | `test_preprocessing_and_prediction_reject_negative_non_numeric_and_oversized` **PASSED** |
| **DEF-04** | **LOW-MEDIUM** | Streamlit UI State (`app/views/prediction_views.py`) | Single prediction widgets lacked unique keys per sample row; batch results vanished upon clicking Download rerun. | Added dynamic widget keys (`single_{init_mode}_{record_ref}_{col}`) and stored batch results in `st.session_state`. | `test_interactive_e2e_single_prediction_history_and_settings` **PASSED** |
| **DEF-05** | **LOW** | UI Layout & Docs (`app/components/theme.py`, `app/views/training_eval_views.py`, `app/views/history_reports_system.py`, `MCA_PBL_PROJECT_REPORT.md`) | Fixed CSS `clamp()` & `min-height` on KPI cards, changed 6-col row to `3x2`, exposed `MCA_PBL_PROJECT_REPORT.md` download, aligned DT/RF training times. | Applied targeted UI & doc fixes. | `48/48` pytest tests **PASSED** |

## Unresolved / Non-Code Remaining Items
1. **LOW / MANUAL:** Live Google Chrome visual inspection (`DevToolsActivePort` inactive on host machine).
2. **INFORMATIONAL:** Student team details and UI screenshots in `README.md`.
