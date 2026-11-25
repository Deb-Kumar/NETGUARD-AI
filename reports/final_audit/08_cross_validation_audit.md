# 08 — Cross-Validation Audit Report

## Verification Evidence
- **Strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` executed strictly on the `175,341`-row training dataset (`test` set completely excluded).
- **Independent Fold Verification:**
  - Fold validation sizes: `[35069, 35068, 35068, 35068, 35068]` (sum = `175,341`, zero overlap across folds).
  - Positive class ratio per fold: `[0.6806, 0.6806, 0.6806, 0.6806, 0.6806]` (exact stratification preserved).
  - Independent arithmetic check confirmed stored `mean` values in `models/metadata/cv_results.json` match fold score averages across all metrics and models.
  - Preprocessing steps (`NetworkFeatureEngineer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`) execute inside each fold pipeline independently.

## Verified 5-Fold Results Summary (Full 175,341 Training Rows)
- **Random Forest:** Recall `0.9531 ± 0.0012`, F1 `0.9654 ± 0.0011`, Precision `0.9781 ± 0.0012`, ROC-AUC `0.9938 ± 0.0003` (`192.57s`)
- **Decision Tree:** Recall `0.9412 ± 0.0025`, F1 `0.9589 ± 0.0008`, Precision `0.9774 ± 0.0020`, ROC-AUC `0.9738 ± 0.0014` (`37.95s`)
- **Logistic Regression:** Recall `0.9513 ± 0.0025`, F1 `0.9504 ± 0.0014`, Precision `0.9495 ± 0.0009`, ROC-AUC `0.9844 ± 0.0006` (`25.27s`)
- **Verdict:** **PASS**
