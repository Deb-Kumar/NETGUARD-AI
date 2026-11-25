# 15 — Database Integrity Audit Report

## SQLite Storage Audited (`src/database/history_db.py`, `database/netguard_history.db`)
- **Schema:** `prediction_history(id, timestamp_utc, prediction_type, model_name, predicted_class, probability, record_reference, notes)` with indexes on `timestamp_utc` and `predicted_class`.
- **Privacy Check:** Sensitive raw IP/port/payload fields are never stored (`record_reference` & summary `notes` only, truncated to safe lengths).
- **CRUD & Filtering Verified:**
  - `init_db()`, `log_prediction()`, `query_history()` (search, `prediction_type`, `predicted_class`, `start_date`/`end_date`, `limit`), `export_history_csv()`, and `clear_history(confirm=True)` (raises `ValueError` when `confirm=False`).
- **Defect Found & Fixed (DEF-01):**
  - *Issue:* `with _connect(db_path) as conn:` did not close `sqlite3.Connection` on exit (`__exit__` only commits/rolls back), causing open file-handle retention and `PermissionError: [WinError 32]` on Windows during temporary directory cleanup.
  - *Resolution:* Wrapped every connection in `contextlib.closing(_connect(db_path))` and verified `TemporaryDirectory` unlinks cleanly on Windows.
- **Verdict:** **PASS**
