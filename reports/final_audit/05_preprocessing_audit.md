# 05 — Preprocessing Validation Audit Report

## Pipeline Architecture Verified
- **Automatic Feature Separation:** `split_feature_columns` distinguishes numeric (`39` base + `8` engineered = `47`) vs categorical (`proto`, `service`, `state`).
- **Estimator-Specific Pipelines:**
  - *Linear (`LogisticRegression`):* `SimpleImputer(strategy='median')` → `StandardScaler()` + categorical `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`.
  - *Tree (`DecisionTree`, `RandomForest`):* `SimpleImputer(strategy='median')` (unscaled) + same categorical encoder.

## Scenario Matrix (Tested Across `Pipeline` & `validate_and_align_features`)

| Scenario | Test Outcome | Details |
|---|---|---|
| A. Complete valid input | **PASS** | Identical predictions between direct pipeline & API (`10/10` exact match) |
| B. Missing numeric & categorical (`NaN`/`None`) | **PASS** | Imputed cleanly with explicit missing-value warning |
| C. Unknown categorical values (`'CLO'`, `'unknown_proto_999'`) | **PASS** | Safely zero-encoded by `OneHotEncoder(handle_unknown='ignore')` |
| D. Invalid numerics (`-5.0`, `'not_a_number'`, `inf`, `1e18`) | **PASS** | `inf`, non-numeric text, and negative values (`< 0`) now raise `PredictionValidationError`; `1e18` processed cleanly |
| E. Missing required column | **PASS** | Raises `PredictionValidationError` naming missing columns |
| F. Extra unnecessary columns | **PASS** | Ignored with warning; never passed to estimator |
| G. Reordered feature columns | **PASS** | Re-aligned strictly to canonical `feature_names` order |
| H. Empty input (`0` rows / empty CSV) | **PASS** | Raises `PredictionValidationError` / `DatasetLoadError` |
| I. Duplicate records | **PASS** | Preserved deterministically during batch scoring |

- **Defect Fixed During Audit:** Strengthened `validate_and_align_features()` to reject negative numbers (`< 0`, verified impossible in UNSW-NB15) and non-numeric strings rather than silently imputing invalid strings.
- **Verdict:** **PASS**
