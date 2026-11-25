# 04 — Data Leakage and Scientific Validity Audit Report

## Leakage Checks Verified
1. **Binary Feature Exclusions (`id`, `attack_cat`, `label`):**
   - Inspected fitted artifact `models/binary/best_model.joblib`: `feature_engineer.input_columns_` contains exactly **42 features**; `['id', 'label', 'attack_cat']` are all absent.
   - Inspected `column_transformer.get_feature_names_out()` (**202 transformed columns**): zero references to `attack_cat`, `label`, or `id`.
   - `validate_no_target_leakage()` raises `DataLeakageError` if forbidden columns are supplied.
2. **Multiclass Independence (`attack_cat`):**
   - Inspected `models/multiclass/best_model.joblib`: `input_columns_` contains strictly the **42 features** (binary `label` and `id` excluded).
3. **Train/Test Pipeline Isolation:**
   - Imputers, `StandardScaler`, and `OneHotEncoder` are inside Scikit-learn `Pipeline` objects fitted exclusively within training folds during CV, GridSearchCV, and final training.
   - Engineered features (`NetworkFeatureEngineer`) are strictly deterministic per single row (e.g. `sbytes + dbytes`, `sttl - dttl`), introducing zero cross-row target/distribution leakage.
4. **Resampling Policy:** No SMOTE or pre-split oversampling applied; `class_weight` was evaluated strictly within training folds.

## Train-Test Signature Overlap & Generalization Investigation
Because the predefined UNSW-NB15 splits drop `srcip`, `sport`, `dstip`, `dsport`, `Stime`, and `Ltime`, `8,541` test rows share identical 42-feature vectors with training rows. To prove the selected Random Forest model is not merely memorizing duplicates, we evaluated `best_model.joblib` separately across both subsets:
- **Overlapping signatures (`n = 8,541`):** Accuracy `0.9521`, Precision `0.9478`, Recall `0.9970`, F1 `0.9718`
- **Strictly unseen feature signatures (`n = 73,791`):** Accuracy `0.8976`, Precision `0.8556`, **Recall `0.9656`**, **F1 `0.9073`**

**Conclusion:** Performance on the `73,791` completely unseen test signatures confirms genuine generalization. Note on `sttl` (`r = +0.6927`) and `ct_state_ttl` (`r = +0.5777`): these are legitimate Argus/Bro flow features generated in the UNSW-NB15 capture environment rather than post-hoc target proxies, though their high predictive weight reflects UNSW-NB15 lab TTL topology.
- **Verdict:** **PASS**
