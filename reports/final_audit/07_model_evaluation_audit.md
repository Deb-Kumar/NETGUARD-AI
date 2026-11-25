# 07 — Model Evaluation Verification Report

## Independent Manual Metric Recalculation
Independently predicted `82,332` untouched test records using `models/binary/best_model.joblib` and verified formulas directly from confusion matrix counts (`1 = Malicious Traffic` positive class, `0 = Normal Traffic` negative class):

- **Observed Confusion Matrix:** `TN = 30,376`, `FP = 6,624`, `FN = 1,338`, `TP = 43,994` (`Sum = 82,332`).
- `Accuracy = (43994 + 30376) / 82332 = 0.903294` → **0.9033** (Matches stored `0.9033`)
- `Precision = 43994 / (43994 + 6624) = 0.869137` → **0.8691** (Matches stored `0.8691`)
- `Recall = 43994 / (43994 + 1338) = 0.970484` → **0.9705** (Matches stored `0.9705`)
- `F1 = 2 * P * R / (P + R) = 0.917019` → **0.9170** (Matches stored `0.9170`)
- `Specificity = 30376 / (30376 + 6624) = 0.820973` → **0.8210** (Matches stored `0.8210`)
- `FPR = 6624 / 37000 = 0.179027` → **0.1790** (Matches stored `0.1790`)
- `FNR = 1338 / 45332 = 0.029516` → **0.0295** (Matches stored `0.0295`)
- `ROC-AUC`: **0.9837** | `PR-AUC`: **0.9877**

## Comparison Table Verification (Untouched Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FNR | Latency (ms/1k) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9033 | 0.8691 | 0.9705 | 0.9170 | 0.9837 | 0.9877 | 0.0295 | 9.051 |
| Decision Tree | 0.8681 | 0.8257 | 0.9640 | 0.8895 | 0.8588 | 0.8176 | 0.0360 | 2.600 |
| Logistic Regression | 0.8351 | 0.8020 | 0.9302 | 0.8613 | 0.9558 | 0.9664 | 0.0698 | 3.639 |

- **Zero-Division Guards:** Verified `zero_division=0` and conditional FPR/Specificity/FNR denominator checks in `src/evaluation/evaluate.py`.
- **Verdict:** **PASS**
