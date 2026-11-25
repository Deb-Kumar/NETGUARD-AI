# 09 — Hyperparameter Tuning Audit Report

## Configuration Audited (`src/training/tuning.py`, `models/metadata/tuning_results.json`)
- **Search Strategy:** `GridSearchCV` with 5-fold `StratifiedKFold(shuffle=True, random_state=42)`.
- **Primary Refit Metric:** `f1` (multi-scoring simultaneously captures `recall`, `precision`, `f1`, `roc_auc`, `accuracy` — balanced choice for intrusion detection without letting majority-class bias sacrifice precision).
- **Training Data Isolation:** Executed on a reproducible stratified `50,000`-row training subset (due to Windows memory/runtime budget of 80 RF fits on `0.69–0.73 GB` free RAM), followed by refitting winning parameters on **100% of the `175,341` training rows**. Test data never entered `GridSearchCV`.

## Parameter Grids & Best Outcomes

| Model | Candidates | Grid Search Time | Full Refit Time | Best Parameters | Best CV F1 |
|---|---:|---:|---:|---|---:|
| Logistic Regression | 12 | `111.86 s` | `44.07 s` | `C=10.0, class_weight=None, solver='liblinear'` | `0.9548` |
| Decision Tree | 24 | `180.65 s` | `15.76 s` | `criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=2` | `0.9563` |
| Random Forest | 16 | `370.41 s` | `102.44 s` | `n_estimators=100, max_depth=25, min_samples_leaf=1, min_samples_split=2` | `0.9632` |

## Findings
- Note: Logistic Regression CV F1 improved from `0.9504` (`class_weight='balanced'`) to `0.9548` (`class_weight=None`) on the 68%-malicious training set, though on the 55%-malicious test set `accuracy` dropped to `0.8351`.
- Sampling approximation (`50,000` out of `175,341`) is explicitly disclosed in `tuning_results.json` and the UI.
- **Verdict:** **PASS**
