# MCA Final-Year PBL Technical Project Report

## NETGUARD AI: AI-Powered Network Intrusion Detection and Threat Classification System Using Machine Learning

---

## 1. Abstract
**NETGUARD AI** is an end-to-end machine learning cybersecurity software product engineered in Python 3.12 to classify network flow records from the **UNSW-NB15** dataset into **Normal Traffic (`0`)** or **Malicious Traffic (`1`)**, extended with an independent **10-class attack-category (`attack_cat`) classifier**. The system enforces strict train/test isolation and target-leakage guards, implements estimator-specific Scikit-learn preprocessing pipelines, benchmarks three algorithms (**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**) via Stratified 5-Fold Cross-Validation and `GridSearchCV`, selects the optimal classifier via a cybersecurity composite validation metric, and exposes a 16-page dark-navy Streamlit web dashboard backed by Joblib model persistence and SQLite prediction logging.

---

## 2. Dataset Inspection & Exploratory Data Analysis Summary

### 2.1 Dataset Partitions Verified
- **Predefined Training Set (`UNSW_NB15_training-set.csv`):** `175,341` records × `45` columns (`62.78 MB` in memory).
  - `label = 1` (Malicious): `119,341` (`68.06%`)
  - `label = 0` (Normal): `56,000` (`31.94%`)
- **Predefined Testing Set (`UNSW_NB15_testing-set.csv`):** `82,332` records × `45` columns (`29.45 MB` in memory).
  - `label = 1` (Malicious): `45,332` (`55.06%`)
  - `label = 0` (Normal): `37,000` (`44.94%`)
- **Raw Full Split Files (`UNSW-NB15_1.csv`–`4.csv`):** Retained untouched for chunked experiments (`49` columns, headerless).

### 2.2 Key Data Quality & Leakage Findings
1. **Zero Missing Cells, Explicit Placeholder `'-'`:** Both predefined partitions contain `0` NaNs, while `service` uses `'-'` (`94,168` training rows; `47,153` testing rows) to mark unspecified services.
2. **Unseen Test Categories:** The testing set `state` column contains `ACC` and `CLO` values absent from the training set, requiring `OneHotEncoder(handle_unknown="ignore")`.
3. **Duplicate & Ambiguous Flow Vectors:** Excluding `id`, `67,601` training rows are duplicates (`74,301` feature-only duplicates, including `229` conflicting feature groups carrying both labels), and `8,541` test rows share identical 42-feature vectors with training rows after removal of IP/port/timestamp fields.
4. **Top Linearly Correlated Features with `label` (Training Set):** `sttl` (`+0.6927`), `ct_state_ttl` (`+0.5777`), `dload` (`-0.3937`), `ct_dst_sport_ltm` (`+0.3572`), and `dmean` (`-0.3418`).

---

## 3. Preprocessing & Leakage Prevention Architecture

- **Explicit Leakage Guards (`src/data/validator.py`):** `validate_no_target_leakage` raises `DataLeakageError` whenever `id`, `attack_cat`, or `label` appears in model feature inputs.
- **Row-Wise Feature Engineering (`NetworkFeatureEngineer`):** Computes `total_bytes`, `total_pkts`, `byte_ratio`, `pkt_ratio`, `avg_bytes_per_pkt`, `ttl_diff`, `loss_total`, and `is_service_unspecified` strictly per single row (zero cross-row statistics).
- **Separate Estimator Pipelines:**
  - *Linear (`Logistic Regression`):* `SimpleImputer(median)` + `StandardScaler()` for `47` numerical features, `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")` for `3` categorical features (`proto`, `service`, `state`).
  - *Tree (`Decision Tree`, `Random Forest`):* `SimpleImputer(median)` (preserving unscaled threshold interpretability) + `OneHotEncoder(handle_unknown="ignore")`.
- **Imbalance Policy:** Training imbalance ratio is `2.131:1` (balanced weights: `0 -> 1.5655`, `1 -> 0.7346`). SMOTE is intentionally avoided to prevent synthetic network-flow artifacts and memory overhead on 175k+ one-hot rows; `class_weight="balanced"` was empirically tested across folds and tuned via `GridSearchCV`.

---

## 4. Experimental Results

### 4.1 Baseline Training Durations (175,341 Training Rows)
- Logistic Regression: `7.207 s`
- Decision Tree Classifier: `11.833 s`
- Random Forest Classifier: `56.946 s`

### 4.2 Stratified 5-Fold Cross-Validation (Full 175,341 Training Rows)

| Model | CV Recall | CV Precision | CV F1 | CV Accuracy | CV ROC-AUC | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| Logistic Regression | `0.9513 ± 0.0025` | `0.9495 ± 0.0009` | `0.9504 ± 0.0014` | `0.9324 ± 0.0019` | `0.9844 ± 0.0006` | `25.27 s` |
| Decision Tree | `0.9412 ± 0.0025` | `0.9774 ± 0.0020` | `0.9589 ± 0.0008` | `0.9451 ± 0.0010` | `0.9738 ± 0.0014` | `37.95 s` |
| **Random Forest** | **`0.9531 ± 0.0012`** | **`0.9781 ± 0.0012`** | **`0.9654 ± 0.0011`** | **`0.9535 ± 0.0015`** | **`0.9938 ± 0.0003`** | `192.57 s` |

### 4.3 GridSearchCV Hyperparameter Tuning
Conducted strictly on training data (5-fold `StratifiedKFold`, primary refit metric `f1`, stratified `50,000`-row tuning subsample with final refit on all `175,341` rows):
- **Logistic Regression** (`12` candidates, `111.86 s` tuning, `44.07 s` full refit): `C=10.0, class_weight=None, solver='liblinear'` → CV `F1=0.9548`, `Recall=0.9882`, `Precision=0.9236`, `ROC-AUC=0.9847`.
- **Decision Tree** (`24` candidates, `180.65 s` tuning, `15.76 s` full refit): `criterion='entropy', max_depth=None, min_samples_leaf=1, min_samples_split=2` → CV `F1=0.9563`, `Recall=0.9562`, `Precision=0.9563`, `ROC-AUC=0.9322`.
- **Random Forest** (`16` candidates, `370.41 s` tuning, `102.44 s` full refit): `n_estimators=100, max_depth=25, min_samples_leaf=1, min_samples_split=2` → CV `F1=0.9632`, `Recall=0.9573`, `Precision=0.9693`, `ROC-AUC=0.9921`.

### 4.4 Validation Selection & Final Untouched Test Evaluation (82,332 Rows)
Composite cybersecurity validation score (`0.30*F1 + 0.30*ROC_AUC + 0.25*Recall + 0.15*Precision`) ranked **Random Forest** #1 (`0.97131` vs Logistic Regression `0.96744` and Decision Tree `0.94905`).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | FNR | FN | FP | Latency (ms/1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| **Random Forest** | **0.9033** | **0.8691** | **0.9705** | **0.9170** | **0.9837** | **0.9877** | **0.8210** | **0.0295** | **1,338** | **6,624** | 9.051 |
| Decision Tree | 0.8681 | 0.8257 | 0.9640 | 0.8895 | 0.8588 | 0.8176 | 0.7507 | 0.0360 | 1,634 | 9,223 | 2.600 |
| Logistic Regression | 0.8351 | 0.8020 | 0.9302 | 0.8613 | 0.9558 | 0.9664 | 0.7186 | 0.0698 | 3,163 | 10,413 | 3.639 |

### 4.5 Independent Multiclass Attack Classification (`attack_cat`)
Trained independently without binary `label` across all `10` classes (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`):
- **Winner:** Random Forest Classifier (`Accuracy=0.6774`, `Macro Precision=0.5162`, `Macro Recall=0.6387`, `Macro F1=0.5029`, `Weighted F1=0.7325`).
- High separability observed on `Generic` (`F1=0.9811`), `Reconnaissance` (`F1=0.8526`), `Normal` (`F1=0.7421`), `Exploits` (`F1=0.7003`), and `Worms` (`F1=0.6786`), while rare/overlapping classes (`Analysis`, `Backdoor`, `DoS`) show lower precision due to shared flow signatures after IP/port removal.

---

## 5. Security, Testing & Quality Assurance
- **Upload & Deserialization Controls:** CSV extension check, `50 MB` limit, schema/type validation, parameterized SQLite queries, and `UntrustedModelError` rejecting non-`.joblib` or out-of-directory model paths.
- **Automated Verification:** Verified via Pytest covering data loading, schema/leakage guards, EDA, preprocessing, training, CV, GridSearchCV, model selection, single/batch inference, SQLite CRUD, multiclass pipeline, and Streamlit AppTest across all 16 UI pages.
