# 02 — Environment and Dependency Audit Report

## Verified Runtime Environment
- **Operating System:** Windows 11 (`Windows-11-10.0.26300-SP0`)
- **Project Python Interpreter:** `D:\Project\NETGUARD-AI\.venv\Scripts\python.exe` — **Python 3.12.10**
- **SQLite Standard Library Version:** `3.49.1`
- **Dependency Integrity Check (`pip check`):** `No broken requirements found.`

## Package Verification (`requirements.txt`)
All required packages imported cleanly inside `.venv`:

| Package | Version (`requirements.txt`) | Import Verified |
|---|---:|---|
| `pandas` | `3.1.1` | Yes |
| `numpy` | `2.5.5` | Yes |
| `scikit-learn` | `1.9.1` | Yes |
| `matplotlib` | `3.11.2` | Yes |
| `seaborn` | `0.13.2` | Yes |
| `plotly` | `7.2.0` | Yes |
| `streamlit` | `1.67.0` | Yes |
| `joblib` | `1.6.0` | Yes |
| `pytest` | `9.2.0` | Yes |
| `sqlite3` (stdlib) | `3.49.1` | Yes |

## Observations
- `pip list --not-required` inside `.venv` also shows `ipykernel==7.4.0` installed locally to support Jupyter notebook execution inside VS Code/IDE; it is not required to run the CLI or Streamlit app.
- Global `python` on `PATH` is Python `3.14.6`; project execution must use `.venv\Scripts\python.exe` (or `py -3.12`).
- **Verdict:** **PASS**
