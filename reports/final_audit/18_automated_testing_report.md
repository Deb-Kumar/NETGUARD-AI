# 18 — Automated Testing Report

## Pytest Suite Structure
All tests reside under `tests/` (`pyproject.toml` `pythonpath=["."]`):
1. `tests/test_environment_and_config.py` (7 tests) — Python 3.12, dependencies, directory layout, dataset resolution, leakage constants, `check-env` CLI
2. `tests/test_data_loader_and_validator.py` (8 tests) — `NUSW-NB15_features.csv` mapping, full train/test profiling, numeric downcasting, raw split streaming, error handling, leakage guards, `inspect-data` CLI
3. `tests/test_eda.py` (2 tests) — numerical/categorical/outlier/correlation/duplicate stats, 8 PNG figures & JSON summary
4. `tests/test_preprocessing.py` (4 tests) — leakage blocking, `NetworkFeatureEngineer`, `linear` vs `tree` pipelines on unseen/missing values, imbalance & config persistence
5. `tests/test_models_training_and_evaluation.py` (2 tests) — end-to-end synthetic `train -> cv -> tune -> select -> evaluate` pipeline + `UntrustedModelError` path/extension enforcement
6. `tests/test_prediction_db_and_multiclass.py` (2 tests) — single prediction, batch CSV (with and without `label`), multiclass pipeline, SQLite CRUD, negative invalid-input rejection
7. `tests/test_security_and_edge_cases.py` (4 tests) — Windows SQLite connection closure (`TemporaryDirectory`), SQL injection safety, CSV formula sanitization, negative/non-numeric/oversized limits, real `best_model.joblib` vs `predict_batch_csv` equivalence
8. `tests/test_streamlit_app.py` (17 tests) — `app/app.py` entrypoint + all 16 view functions via `AppTest`
9. `tests/test_e2e_user_journey.py` (2 tests) — interactive `AppTest` E2E workflow (Single Prediction -> SQLite History -> Clear Confirmation + Settings session state)

## Final Execution Result (After All Fixes)
- **Command:** `D:\Project\NETGUARD-AI\.venv\Scripts\python.exe -m pytest`
- **Outcome:** **48 passed, 0 failed, 0 skipped** in `20.36s`.
- **Note on Warnings:** 13 benign Seaborn/Matplotlib deprecation warnings originating inside `seaborn==0.13.2` when paired with `matplotlib==3.11.2`.
- **Verdict:** **PASS**
