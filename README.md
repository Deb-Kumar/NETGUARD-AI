# NETGUARD AI

**AI-Powered Network Intrusion Detection and Threat Classification System Using Machine Learning**

---

## 1. Project Introduction

**NETGUARD AI** is an MCA Final-Year Project-Based Learning (PBL) cybersecurity machine learning application built on **Python 3.12**, **Scikit-learn**, **SQLite**, and **Streamlit**. It analyzes flow-level network traffic records from the **UNSW-NB15** dataset to classify flows into **Normal Traffic (`0`)** or **Malicious Traffic (`1`)**, alongside an independent **10-class attack-category (`attack_cat`) classifier**.

> **Operational Scope Notice:** NETGUARD AI is an offline dataset- and CSV-driven intrusion classification platform. It does **not** perform real-time packet capture or live network monitoring, and automated ML predictions do **not** guarantee the presence or absence of an actual cyberattack.

---

## 2. Project Objectives

1. Inspect, validate, and profile the UNSW-NB15 dataset while keeping original raw files unmodified.
2. Perform comprehensive Exploratory Data Analysis (12 statistical analyses and 8 visual reports).
3. Build leakage-free, estimator-appropriate Scikit-learn `ColumnTransformer` preprocessing pipelines with row-wise cybersecurity feature engineering and `OneHotEncoder(handle_unknown="ignore")`.
4. Train, cross-validate (`StratifiedKFold`, 5 folds), tune (`GridSearchCV`), and evaluate three classifiers:
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
5. Select the best binary model using a non-redundant composite cybersecurity score computed strictly on validation folds (prioritizing Attack F1, ROC-AUC, Attack Recall, and Precision rather than raw accuracy alone), then evaluate once on the untouched test partition.
6. Provide an independent 10-class `attack_cat` extension without binary `label` leakage.
7. Deliver a 16-page dark-navy Streamlit cybersecurity dashboard supporting single-record prediction, validated batch CSV classification, SQLite prediction history, and downloadable reports.

---

## 3. Problem Statement

Signature-based intrusion detection systems struggle against modern multi-stage and polymorphic attacks. Analyzing statistical network flow telemetry (protocols, services, connection states, byte/packet asymmetry, TTL behavior, jitter, and connection tracking counts) with supervised ML allows analysts to screen suspicious traffic batches efficiently while monitoring false negative rates (`FNR`).

---

## 4. Dataset Information

Source: **UNSW-NB15** (Australian Centre for Cyber Security — Moustafa & Slay, 2015/2016).

- **Predefined Training Set (`UNSW_NB15_training-set.csv`):** `175,341` records × `45` columns (`119,341` Malicious [`68.06%`], `56,000` Normal [`31.94%`], `0` missing cells).
- **Predefined Testing Set (`UNSW_NB15_testing-set.csv`):** `82,332` records × `45` columns (`45,332` Malicious [`55.06%`], `37,000` Normal [`44.94%`], `0` missing cells).
- **4 Raw Split CSVs (`UNSW-NB15_1.csv`–`4.csv`):** Retained untouched for chunked experiments (`49` columns, headerless).
- **Schema Notes Verified During Inspection:**
  - `NUSW-NB15_features.csv` requires `latin1` encoding and describes the 49 raw columns; predefined sets drop IP/port/timestamp fields (`srcip`, `sport`, `dstip`, `dsport`, `Stime`, `Ltime`), add `id` and `rate`, and normalize column names.
  - `service` uses `'-'` as an explicit unspecified placeholder (`94,168` training rows; `47,153` test rows).
  - Test `state` contains unseen categories `['ACC', 'CLO']` absent from training data (`handle_unknown="ignore"` handles these cleanly).
  - `id`, `attack_cat`, and `label` are strictly excluded from binary input features via `validate_no_target_leakage`.

---

## 5. System Architecture

```text
UNSW-NB15 CSVs (Read-Only)
   │
   ▼
[src/data/] loader.py + validator.py + eda.py
   │  (Encoding fallback, chunking, DataLeakageError guards, EDA reports/figures)
   ▼
[src/preprocessing/] pipeline.py
   │  (NetworkFeatureEngineer + ColumnTransformer: median/most_frequent,
   │   StandardScaler for linear models only, OneHotEncoder(handle_unknown="ignore"))
   ▼
[src/models/ & src/training/] train.py + cross_validation.py + tuning.py + selector.py + multiclass.py
   │  (5-Fold Stratified CV & GridSearchCV on training data only; multi-criteria selector)
   ▼
[models/ & src/evaluation/] Joblib Pipelines + JSON Metadata + Confusion/ROC/PR Evaluation
   │
   ▼
[src/prediction/ & src/database/] Single/Batch Inference + Parameterized SQLite History
   │
   ▼
[app/] 16-Page Dark-Navy Streamlit Cybersecurity Dashboard
```

---

## 6. Technology Stack

- **Python:** `3.12.10`
- **Data Processing:** `pandas==3.1.1`, `numpy==2.5.5`
- **Machine Learning & Persistence:** `scikit-learn==1.9.1`, `joblib==1.6.0`
- **Visualization:** `matplotlib==3.11.2`, `seaborn==0.13.2`, `plotly==7.2.0`
- **Web Interface & Storage:** `streamlit==1.67.0`, `sqlite3` (Python stdlib `3.49.1`)
- **Testing & VCS:** `pytest==9.2.0`, Git

---

## 7. Project Structure

```text
NETGUARD-AI/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── components/
│   │   ├── charts.py
│   │   ├── data_access.py
│   │   └── theme.py
│   └── views/
│       ├── cv_tuning_comparison.py
│       ├── dataset_views.py
│       ├── eda_preprocessing.py
│       ├── history_reports_system.py
│       ├── overview.py
│       ├── prediction_views.py
│       └── training_eval_views.py
├── database/
├── dataset/
│   ├── processed/
│   └── raw/
├── models/
│   ├── binary/
│   ├── metadata/
│   └── multiclass/
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── reports/
│   ├── figures/
│   ├── MCA_PBL_PROJECT_REPORT.md
│   ├── dataset_inspection_report.json
│   └── eda_summary.json
├── src/
│   ├── data/
│   │   ├── eda.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── database/
│   │   └── history_db.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── plots.py
│   ├── models/
│   │   ├── base.py
│   │   ├── decision_tree.py
│   │   ├── logistic_regression.py
│   │   └── random_forest.py
│   ├── prediction/
│   │   └── predict.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── training/
│   │   ├── cross_validation.py
│   │   ├── multiclass.py
│   │   ├── selector.py
│   │   ├── train.py
│   │   └── tuning.py
│   └── utils/
│       └── config.py
├── tests/
├── .gitignore
├── main.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 8. Installation Instructions (Windows)

```powershell
cd D:\Project\NETGUARD-AI
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py check-env
```

---

## 9. Dataset Setup Instructions

Keep the official UNSW-NB15 files (`Training and Testing Sets/UNSW_NB15_training-set.csv`, `Training and Testing Sets/UNSW_NB15_testing-set.csv`, `NUSW-NB15_features.csv`, and `UNSW-NB15_1.csv`–`4.csv`) in the project root or `dataset/raw/`. Validate discovery and integrity:

```powershell
python main.py inspect-data --save-report
python main.py run-eda
```

---

## 10. Model Training & Tuning Instructions

```powershell
# Train baseline binary pipelines + class_weight study
python main.py train

# Run 5-fold Stratified Cross-Validation on full training set (175,341 rows)
python main.py cross-validate --n-splits 5

# Run GridSearchCV on 50,000 stratified training subset & refit on all 175,341 rows
python main.py tune --cv-folds 5 --scoring f1 --max-tuning-samples 50000

# Select best model from validation scores & evaluate once on untouched test set
python main.py select-best --strategy cybersecurity_composite

# Train & evaluate independent 10-class attack_cat classifier
python main.py train-multiclass
```

---

## 11. Evaluation & Testing Instructions

```powershell
# Evaluate saved binary models against untouched test set
python main.py evaluate

# Run complete automated pytest suite
pytest
```

---

## 12. Application Execution Instructions

```powershell
.venv\Scripts\activate
streamlit run app/app.py
```

---

## 13. Experimental Results (Actual Executed Runs)

### A. 5-Fold Stratified Cross-Validation (Full Training Set: 175,341 Rows)

| Model | CV Recall (Mean ± Std) | CV Precision Mean | CV F1 (Mean ± Std) | CV Accuracy Mean | CV ROC-AUC Mean | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| **Random Forest Classifier** | **0.9531 ± 0.0012** | **0.9781** | **0.9654 ± 0.0011** | **0.9535** | **0.9938** | 192.57 s |
| Decision Tree Classifier | 0.9412 ± 0.0025 | 0.9774 | 0.9589 ± 0.0008 | 0.9451 | 0.9738 | 37.95 s |
| Logistic Regression | 0.9513 ± 0.0025 | 0.9495 | 0.9504 ± 0.0014 | 0.9324 | 0.9844 | 25.27 s |

### B. Untouched Test Set Evaluation (82,332 Rows, Tuned & Refitted on Full 175,341 Training Rows)

**Selected Winner (`cybersecurity_composite` validation score `0.97131`):** **Random Forest Classifier** (`n_estimators=100`, `max_depth=25`, `min_samples_split=2`, `min_samples_leaf=1`).

| Model | Accuracy | Attack Precision | Attack Recall | Attack F1 | ROC-AUC | PR-AUC | Specificity | FNR | Latency (ms / 1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| **Random Forest Classifier** | **0.9033** | **0.8691** | **0.9705** | **0.9170** | **0.9837** | **0.9877** | **0.8210** | **0.0295** | 9.051 |
| Decision Tree Classifier | 0.8681 | 0.8257 | 0.9640 | 0.8895 | 0.8588 | 0.8176 | 0.7507 | 0.0360 | 2.600 |
| Logistic Regression | 0.8351 | 0.8020 | 0.9302 | 0.8613 | 0.9558 | 0.9664 | 0.7186 | 0.0698 | 3.639 |

- **Random Forest Test Confusion Matrix (`N = 82,332`):** `TN = 30,376`, `FP = 6,624`, `FN = 1,338`, `TP = 43,994`.

### C. Independent Multiclass Classification (`attack_cat`, 10 Classes, Untouched Test Set)

- **Selected Model:** Random Forest Classifier (`macro_f1 = 0.5029`, `weighted_f1 = 0.7325`, `accuracy = 0.6774`, `macro_recall = 0.6387`, `macro_precision = 0.5162`, latency `9.775 ms / 1k`).

---

## 14. Visual Reports Generated (`reports/figures/`)

1. `01_binary_class_distribution.png` — Binary Normal vs Malicious class distribution
2. `02_attack_category_distribution.png` — 10-class `attack_cat` distribution
3. `03_protocol_distribution.png` — Top 10 protocols by traffic class
4. `04_service_distribution.png` — Top services by traffic class (including `'-'`)
5. `05_numerical_feature_distributions.png` — `log1p` feature distributions (30,000 stratified sample)
6. `06_outlier_boxplots.png` — Class-wise outlier box plots (20,000 stratified sample)
7. `07_correlation_heatmap.png` — Correlation heatmap of top 15 features vs `label`
8. `08_state_distribution.png` — Connection states by traffic class
9. `09_confusion_matrices_binary.png` — Untouched test confusion matrices across all 3 models
10. `10_roc_curves_binary.png` — Overlaid ROC curves
11. `11_pr_curves_binary.png` — Overlaid Precision-Recall curves
12. `12_model_metrics_comparison.png` — Multi-metric cybersecurity bar comparison
13. `13_multiclass_confusion_matrix.png` — 10-class row-normalized confusion matrix

*(Dashboard screenshots can be captured in `reports/` after launching `streamlit run app/app.py` for MCA submission.)*

---

## 15. Limitations

- Offline classification on flow records/CSVs only; no live NIC packet sniffing.
- Dropping IP/port/timestamp columns in the predefined UNSW-NB15 splits leaves `229` conflicting feature groups in training data and `8,541` feature-signature overlaps between train and test, limiting single-flow separability on minority multiclass categories (`Analysis`, `Backdoor`, `DoS`).
- Predictions do not certify actual intrusion occurrence or safety in unseen enterprise networks.

---

## 16. Future Enhancements

- Real-time packet/flow collector adapter (PCAP/Zeek/Argus streaming).
- Per-prediction explainability (feature importance / SHAP summaries) and drift detection.
- Multi-user authentication and SIEM webhook alerting.

---

## 17. Team Information

- **Project Type:** MCA Final-Year Project-Based Learning (PBL)
- **Student / Team Members:** *[Insert Name(s) and Roll Number(s)]*
- **Faculty Guide:** *[Insert Faculty Guide Name]*
- **Department & Institution:** *[Insert Department & University Name]*
