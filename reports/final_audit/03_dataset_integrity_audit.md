# 03 — Dataset Integrity Audit Report

## Inspected Dataset Files
All original UNSW-NB15 dataset files remain untouched at their source locations:
- `Training and Testing Sets/UNSW_NB15_training-set.csv`: **175,341 rows × 45 columns** (`32,293,018` bytes, `utf-8-sig`)
- `Training and Testing Sets/UNSW_NB15_testing-set.csv`: **82,332 rows × 45 columns** (`15,380,800` bytes, `utf-8-sig`)
- `UNSW-NB15_1.csv`–`4.csv`: 49-column headerless raw split files (`700,001`, `700,001`, `700,001`, `440,044` lines) retained separately and never mixed silently with the 45-column predefined partitions.
- `NUSW-NB15_features.csv`: **49 rows**, `latin1` encoding, mapped cleanly to all 45 predefined columns (`smeansz`→`smean`, `dmeansz`→`dmean`, `res_bdy_len`→`response_body_len`, `ct_src_ ltm`→`ct_src_ltm`, plus `id` and `rate`).

## Integrity Findings
- **Missing Cells:** `0` NaN values in both predefined training and testing sets; `service='-'` (`94,168` train / `47,153` test) is handled explicitly as an informative placeholder and engineered indicator.
- **Infinite / Invalid Numerics:** `0` `±inf` values across all numerical columns; verified minimum value across all 39 base numerical features is `>= 0.0`.
- **Target Mapping (`label`):** `0 = Normal Traffic`, `1 = Malicious Traffic`. Zero mismatches between `label == 0` and `attack_cat == 'Normal'` in both train and test.
  - Train: `1` = `119,341` (`68.06%`), `0` = `56,000` (`31.94%`)
  - Test: `1` = `45,332` (`55.06%`), `0` = `37,000` (`44.94%`)
- **Attack Categories (`attack_cat`):** 10 verified classes match identically across train and test (`Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`).
- **Unseen Categorical Values:** Test `state` contains `['ACC', 'CLO']` absent from train; handled safely via `OneHotEncoder(handle_unknown="ignore")`.
- **Duplicates:** `0` exact row duplicates (with `id`); `67,601` train / `26,387` test duplicates excluding `id`; `229` conflicting feature-only duplicate groups in train.
- **Verdict:** **PASS**
