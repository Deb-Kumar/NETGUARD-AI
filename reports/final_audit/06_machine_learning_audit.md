# 06 — Machine Learning Model Audit Report

## Inspected Algorithms (Binary & Multiclass)

### A. Logistic Regression (`models/binary/logistic_regression.joblib`)
- **Tuned Parameters:** `C=10.0`, `solver='liblinear'`, `class_weight=None`, `max_iter=1000`, `random_state=42`.
- **Scaling:** Paired with `StandardScaler()` (`linear` pipeline).
- **Observations:** Converged cleanly without warnings (`7.21s` baseline fit; `44.07s` full refit). Unweighted `class_weight=None` slightly favored the `68%` malicious training prior (`98.82%` CV recall), yielding `83.51%` accuracy and `93.02%` recall on the more balanced `55:45` test partition.

### B. Decision Tree Classifier (`models/binary/decision_tree.joblib`)
- **Tuned Parameters:** `criterion='entropy'`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight='balanced'`, `random_state=42`.
- **Observations:** Fast inference (`2.600 ms / 1k rows`), `86.81%` test accuracy, `96.40%` attack recall, `88.95%` F1. Unbounded depth (`max_depth=None`) reduces probability calibration (`ROC-AUC=0.8588`), making ensemble Random Forest clearly superior.

### C. Random Forest Classifier (`models/binary/random_forest.joblib` & `best_model.joblib`)
- **Tuned Parameters:** `n_estimators=100`, `max_depth=25`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight='balanced'`, `n_jobs=2`, `random_state=42`.
- **Observations:** `56.95s` baseline fit, `102.44s` tuned refit, `15.35 MB` compressed artifact, `9.051 ms / 1k rows` evaluation latency, `90.33%` test accuracy, `97.05%` attack recall, `98.37%` ROC-AUC.

### D. Multiclass Extension (`models/multiclass/best_model.joblib`)
- **Model:** Independent Random Forest (`class_weight='balanced'`), 10 `attack_cat` classes discovered from dataset, `67.74%` test accuracy, `0.7325` weighted F1, `0.5029` macro F1.
- **Verdict:** **PASS**
