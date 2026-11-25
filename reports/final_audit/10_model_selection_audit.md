# 10 — Best Model Selection Audit Report

## Selection Logic Inspected (`src/training/selector.py`)
- **Strategy (`cybersecurity_composite`):**
  $$\text{Score} = 0.30 \cdot \text{CV\_F1} + 0.30 \cdot \text{CV\_ROC\_AUC} + 0.25 \cdot \text{CV\_Recall} + 0.15 \cdot \text{CV\_Precision}$$
- **Strict Separation:** Validation ranking relies exclusively on `GridSearchCV` / `StratifiedKFold` validation metrics. Untouched test predictions are computed only **after** ranking for final reporting.

## Validation Ranking (No Test Data Used)

| Rank | Model | Val F1 | Val ROC-AUC | Val Recall | Val Precision | Composite Score |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **Random Forest Classifier** | **0.9632** | **0.9921** | **0.9573** | **0.9693** | **0.97131** |
| 2 | Logistic Regression | 0.9548 | 0.9847 | 0.9882 | 0.9236 | 0.96744 |
| 3 | Decision Tree Classifier | 0.9563 | 0.9322 | 0.9562 | 0.9563 | 0.949045 |

## Contamination Assessment
- Earlier during Phase 6–10 development, a mathematical redundancy (`0.35*recall + 0.20*(1 - fnr)` double-counting `recall == 1 - fnr` while omitting `roc_auc`) had initially selected Logistic Regression. Fixing that mathematical redundancy from first principles properly ranked Random Forest #1 on validation folds, aligning with its `0.9837` test ROC-AUC, `0.9705` test recall, and `0.9170` test F1. No test-label hyperparameter fitting or test-driven threshold tuning occurred.
- **Verdict:** **PASS**
