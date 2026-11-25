"""Phase 4 tests: Exploratory Data Analysis and report figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.eda import (
    compute_categorical_summary,
    compute_correlation_analysis,
    compute_duplicate_conflict_analysis,
    compute_numerical_summary,
    compute_outlier_summary,
    run_eda_pipeline,
)
from src.data.validator import PREDEFINED_COLUMNS
from src.utils.config import get_paths


def _make_sample_df(rows: int = 30) -> pd.DataFrame:
    data: dict[str, list[object]] = {}
    for col in PREDEFINED_COLUMNS:
        if col == "id":
            data[col] = list(range(1, rows + 1))
        elif col == "proto":
            data[col] = ["tcp" if i % 2 == 0 else "udp" for i in range(rows)]
        elif col == "service":
            data[col] = ["-" if i % 3 == 0 else "http" for i in range(rows)]
        elif col == "state":
            data[col] = ["FIN" if i % 2 == 0 else "INT" for i in range(rows)]
        elif col == "attack_cat":
            data[col] = ["Normal" if i % 2 == 0 else "Generic" for i in range(rows)]
        elif col == "label":
            data[col] = [0 if i % 2 == 0 else 1 for i in range(rows)]
        else:
            val_fn = lambda i: float((i % 7) + (100 if i == rows - 1 else 1))
            data[col] = [val_fn(i) for i in range(rows)]
    return pd.DataFrame(data)


def test_eda_summary_computations() -> None:
    df = _make_sample_df(24)
    num_sum = compute_numerical_summary(df)
    assert "dur" in num_sum
    assert {"mean", "std", "min", "median", "max", "skewness", "zero_pct"}.issubset(
        num_sum["dur"].keys()
    )

    cat_sum = compute_categorical_summary(df)
    assert set(cat_sum.keys()) == {"proto", "service", "state"}
    assert cat_sum["service"]["dash_count"] == 8

    outliers = compute_outlier_summary(df)
    assert "dur" in outliers
    assert outliers["dur"]["outlier_count"] >= 1

    corr = compute_correlation_analysis(df, top_n=5)
    assert len(corr["top_features"]) == 5

    dupes = compute_duplicate_conflict_analysis(df)
    assert dupes["exact_duplicates"] == 0

    # Clone row 0 features with opposite label to verify conflict detection
    conflict_df = pd.concat([df, df.iloc[[0]].copy()], ignore_index=True)
    conflict_df.loc[len(conflict_df) - 1, "id"] = 999
    conflict_df.loc[len(conflict_df) - 1, "label"] = 1
    conflict_df.loc[len(conflict_df) - 1, "attack_cat"] = "Generic"
    conflict_dupes = compute_duplicate_conflict_analysis(conflict_df)
    assert conflict_dupes["feature_only_duplicates"] >= 1
    assert conflict_dupes["conflicting_feature_groups"] == 1


def test_run_eda_pipeline_writes_figures_and_json(tmp_path: Path) -> None:
    train_df = _make_sample_df(40)
    test_df = _make_sample_df(20)
    temp_paths = get_paths(tmp_path)

    result = run_eda_pipeline(
        train_df=train_df,
        test_df=test_df,
        paths=temp_paths,
        generate_plots=True,
    )
    summary_path = Path(result["summary_path"])
    assert summary_path.is_file()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(loaded["generated_figures"]) == 8
    for fig_path in loaded["generated_figures"].values():
        p = Path(fig_path)
        assert p.is_file() and p.stat().st_size > 1000
