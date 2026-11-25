# 13 — Batch CSV Prediction Audit Report

## Scenarios Tested (`predict_batch_csv` & `tests/`)

| # | Scenario | Result |
|---|---|---|
| 1 | Correct CSV file | **PASS** — outputs predictions, probabilities, normal/malicious counts, latency |
| 2 | Empty CSV file | **PASS** — raises `PredictionValidationError` |
| 3 | Incorrect file extension (`.exe`) | **PASS** — raises `PredictionValidationError` |
| 4 | Missing required feature columns | **PASS** — raises `PredictionValidationError` listing missing columns |
| 5 | Extra columns | **PASS** — ignored safely with warning |
| 6 | Reordered columns | **PASS** — aligned automatically to canonical `feature_names` |
| 7 | Missing numeric/categorical values | **PASS** — imputed cleanly via fitted pipeline with warning |
| 8 | Unknown categorical values (`UNSEEN_STATE`) | **PASS** — handled via `OneHotEncoder(handle_unknown='ignore')` |
| 9 | Duplicate records | **PASS** — scored consistently row-by-row |
| 10 | Large file / chunked inference & >50 MB check | **PASS** — chunked (`chunksize=25_000`) + rejects `>50 MB`; `file_size_bytes` override defect fixed |
| 11 | Negative or non-numeric numeric cells | **PASS** — rejects negative/unparseable values with `PredictionValidationError` |
| 12 | CSV containing actual `label` ground truth | **PASS** — computes optional metrics with explicit disclaimer |
| 13 | Prediction-only CSV without `label` | **PASS** — `has_ground_truth=False`, skips evaluation cleanly |

- **UI Persistence Fix:** Stored `BatchPredictionResult` in `st.session_state` so clicking "Download Full Predictions CSV" does not wipe displayed metrics/charts or double-log SQLite rows. Formula-prefix sanitization applied to downloaded CSV cells.
- **Verdict:** **PASS**
