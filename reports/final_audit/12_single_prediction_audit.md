# 12 — Single Record Prediction Audit Report

## Verified Behavior (`src/prediction/predict.py`, `app/views/prediction_views.py`)
- **Required Output Fields Verified:** `predicted_class` (`Normal Traffic` / `Malicious Traffic`), `model_name`, `timestamp_utc`, `malicious_probability`, `normal_probability`, optional `multiclass_attack_category` + `multiclass_confidence`, and explicit advisory disclaimer stating predictions do not guarantee actual attacks.
- **Validation & Negative Cases Verified:**
  - Empty dict `{}` → raises `PredictionValidationError`
  - Missing required columns → raises `PredictionValidationError`
  - Infinite values (`±inf`) → raises `PredictionValidationError`
  - Negative numerical values (`dur=-1.0`) → raises `PredictionValidationError` (**Fixed during audit**)
  - Non-numeric string values (`sbytes="invalid_text"`) → raises `PredictionValidationError` (**Fixed during audit**)
  - Explicit `None`/`NaN` optional numeric/categorical values → handled safely via pipeline imputation with warning
  - Shuffled dictionary key order → aligned cleanly to canonical 42-feature order
  - Extreme float (`1e18`) → handled without overflow crash
- **UI State Fix Verified:** Added dynamic widget `key`s (`single_{init_mode}_{record_ref}_{col_name}`) so switching test sample records in the Streamlit UI refreshes all 42 form inputs cleanly.
- **Verdict:** **PASS**
