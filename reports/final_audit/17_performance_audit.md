# 17 — Performance and Resource Audit Report

## Test Machine Specifications
- **OS:** Windows 11 (`Windows-11-10.0.26300-SP0`)
- **CPU:** 16 logical cores (`DEFAULT_N_JOBS` bounded to `2` to prevent memory contention)
- **RAM:** `15.68 GB` total (`0.69–0.73 GB` free physical RAM observed during heavy runs)
- **Dataset:** UNSW-NB15 predefined training (`175,341` rows) & testing (`82,332` rows)

## Measured Performance Benchmarks

| Operation | Dataset Size | Measured Duration / Throughput | Memory / Artifact Size |
|---|---|---:|---|
| CSV Loading (`train` + `test`) | `257,673` rows | `0.297 s` | `62.78 MB` train + `29.45 MB` test |
| Preprocessing (`ColumnTransformer` transform) | `82,332` rows (`202` sparse cols) | `0.808 s` | `146.81 MB` peak traced RAM |
| Baseline Training (`LR` / `DT` / `RF`) | `175,341` rows | `7.21 s` / `11.83 s` / `56.95 s` | `0.01 MB` / `0.21 MB` / `14.84 MB` |
| 5-Fold Stratified CV (`LR` / `DT` / `RF`) | `175,341` rows | `25.27 s` / `37.95 s` / `192.57 s` | Bounded (`n_jobs=1` outer) |
| GridSearchCV + Full Refit (`LR` / `DT` / `RF`) | `50,000` search + `175,341` refit | `155.9 s` / `196.4 s` / `472.8 s` | Safe under low free RAM |
| Inference Throughput (Small) | `1,000` rows | `102.07 ms` (`9,798 rows/s`) | Uses cached `best_model.joblib` (`15.35 MB`) |
| Inference Throughput (Medium) | `25,000` rows | `601.11 ms` (`41,590 rows/s`) | Chunked (`chunksize=25_000`) |
| Inference Throughput (Full Test Set) | `82,332` rows | `1,876.64 ms` (`43,872 rows/s`) | Full test scoring `< 2 s` |

## Caching & Reruns Verification
- `@st.cache_data` caches JSON metadata, bounded dataset previews (`nrows<=10,000`), and feature dictionaries cleanly without retraining during UI interactions.
- **Verdict:** **PASS**
