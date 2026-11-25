# 11 — Model Persistence Audit Report

## Artifact Inventory Verified
- `models/binary/best_model.joblib` (`15.35 MB`)
- `models/binary/random_forest.joblib` (`14.84 MB`)
- `models/binary/decision_tree.joblib` (`0.21 MB`)
- `models/binary/logistic_regression.joblib` (`0.01 MB`)
- `models/multiclass/best_model.joblib` (`19.54 MB`), `decision_tree.joblib` (`0.28 MB`), `random_forest.joblib` (`19.54 MB`)
- Metadata JSONs: `model_metadata.json`, `multiclass_metadata.json`, `binary_training_summary.json`, `binary_evaluation_results.json`, `cv_results.json`, `tuning_results.json`, `preprocessing_config.json`.

## Verification Tests Executed
1. **Prediction Equivalence:** Verified `np.array_equal(best_model.predict(X_test), random_forest.predict(X_test))` across all `82,332` test rows, and verified `best_model.predict()` matches `predict_batch_csv()` identically on sampled rows.
2. **Full Pipeline Serialization:** Confirmed serialized `.joblib` objects store complete `Pipeline([("preprocessor", ...), ("classifier", ...)])` including `NetworkFeatureEngineer`, `SimpleImputer`, `StandardScaler` (for LR), and `OneHotEncoder(handle_unknown='ignore')`.
3. **Trust & Error Handling:** Verified `load_model_artifact()` raises `UntrustedModelError` for files outside `models/` and non-`.joblib` extensions, and `FileNotFoundError` for missing models.
- **Verdict:** **PASS**
