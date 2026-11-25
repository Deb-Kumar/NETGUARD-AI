# 16 — Defensive Security Audit Report

## Security Controls Audited & Tested

| Control Area | Verification Outcome | Evidence / Notes |
|---|---|---|
| Uploaded File Type & Size | **PASS** | Rejects non-`.csv` files (`.exe`, etc.) and uploads exceeding `50 MB` (`MAX_UPLOAD_SIZE_BYTES`, fixed `file_size_bytes` overwrite bug) |
| Uploaded File Execution | **PASS** | CSVs parsed purely as passive data via `pandas.read_csv`; never executed or written to executable paths |
| Path Traversal & Untrusted Models | **PASS** | `load_model_artifact()` resolves paths against `models/` via `relative_to()` and requires `.joblib`, blocking outside/pickle paths (`UntrustedModelError`); `load_json_artifact()` also verifies `relative_to(root)` |
| SQL Injection | **PASS** | All SQL queries in `history_db.py` use `?` placeholders; tested `"' OR '1'='1; DROP TABLE prediction_history; --"` safely returning `0` rows without table impact |
| CSV Formula Injection | **PASS (Fixed)** | Added `sanitize_csv_cell` & `sanitize_dataframe_for_csv` neutralizing `=`, `+`, `@`, `\t`, `\r`, and non-numeric `-` prefixes in both history exports and batch CSV downloads |
| Secrets / Credentials | **PASS** | Regex scan confirmed zero hardcoded API keys, tokens, or passwords; `.env` & `secrets.toml` excluded in `.gitignore` |
| Error Handling & Path Exposure | **PASS** | Validation errors return clean user-facing messages (`PredictionValidationError`, `DatasetLoadError`); UI shows filenames rather than host directories |
| Dependency Integrity | **PASS** | `pip check` reports zero broken dependencies; Python `3.12.10` + `OpenSSL 3.0.16` |

- **Scope Note:** Designed as a single-user local academic analytics tool without authentication or multi-tenant deployment exposure.
- **Verdict:** **PASS**
