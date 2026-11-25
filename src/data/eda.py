"""Exploratory Data Analysis (EDA) and visualization generator for UNSW-NB15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import (
    load_training_and_testing_sets,
    profile_dataset,
    stratified_sample_df,
)
from src.data.validator import (
    PREDEFINED_CATEGORICAL_COLUMNS,
    PREDEFINED_FEATURE_COLUMNS,
    PREDEFINED_NUMERICAL_COLUMNS,
)
from src.utils.config import (
    BINARY_TARGET_COLUMN,
    ID_COLUMN,
    MULTICLASS_TARGET_COLUMN,
    ProjectPaths,
    RANDOM_STATE,
    ensure_directories,
    get_paths,
)

# Palette aligned with NETGUARD AI dark cybersecurity theme
COLOR_NORMAL = "#10B981"  # Green
COLOR_MALICIOUS = "#EF4444"  # Red
COLOR_ACCENT = "#3B82F6"  # Blue
COLOR_WARN = "#F59E0B"  # Amber

KEY_DISTRIBUTION_FEATURES: tuple[str, ...] = (
    "dur",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "ct_state_ttl",
)


def _configure_plot_style() -> None:
    """Apply consistent, publication-ready Matplotlib/Seaborn styling."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def compute_numerical_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics, skewness, and zero ratios for numerical features."""
    summary: dict[str, dict[str, float]] = {}
    num_cols = [c for c in PREDEFINED_NUMERICAL_COLUMNS if c in df.columns]
    desc = df[num_cols].describe().T
    skewness = df[num_cols].skew(numeric_only=True)
    n_rows = max(len(df), 1)

    for col in num_cols:
        row = desc.loc[col]
        zero_pct = float((df[col] == 0).sum()) / n_rows * 100.0
        summary[col] = {
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 4),
            "std": round(float(row["std"]), 4),
            "min": round(float(row["min"]), 4),
            "q25": round(float(row["25%"]), 4),
            "median": round(float(row["50%"]), 4),
            "q75": round(float(row["75%"]), 4),
            "max": round(float(row["max"]), 4),
            "skewness": round(float(skewness.get(col, 0.0)), 4),
            "zero_pct": round(zero_pct, 2),
        }
    return summary


def compute_categorical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize cardinality, top categories, and attack rates for categorical features."""
    result: dict[str, dict[str, Any]] = {}
    for col in PREDEFINED_CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts()
        top_10 = {str(k): int(v) for k, v in counts.head(10).items()}
        attack_rates: dict[str, float] = {}
        if BINARY_TARGET_COLUMN in df.columns:
            top_keys = list(top_10.keys())
            subset = df[df[col].astype(str).isin(top_keys)]
            grouped = subset.groupby(subset[col].astype(str))[BINARY_TARGET_COLUMN].mean()
            attack_rates = {str(k): round(float(v) * 100.0, 2) for k, v in grouped.items()}
        result[col] = {
            "unique_count": int(counts.shape[0]),
            "top_10_counts": top_10,
            "top_10_attack_rate_pct": attack_rates,
            "dash_count": int((df[col].astype(str).str.strip() == "-").sum()),
        }
    return result


def compute_outlier_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Analyze IQR-based outliers across numerical features."""
    outliers: dict[str, dict[str, float | int]] = {}
    num_cols = [c for c in PREDEFINED_NUMERICAL_COLUMNS if c in df.columns]
    n_rows = max(len(df), 1)
    for col in num_cols:
        series = df[col].astype(float)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outliers[col] = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": count,
            "outlier_pct": round(count / n_rows * 100.0, 2),
        }
    return outliers


def compute_correlation_analysis(
    df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> dict[str, Any]:
    """Compute feature correlations with binary target on training data only."""
    num_cols = [c for c in PREDEFINED_NUMERICAL_COLUMNS if c in df.columns]
    if BINARY_TARGET_COLUMN not in df.columns:
        return {"target_correlations": {}, "top_features": num_cols[:top_n]}

    corr_with_target = (
        df[num_cols]
        .corrwith(df[BINARY_TARGET_COLUMN].astype(float))
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    top_features = list(corr_with_target.head(top_n).index)
    corr_matrix = df[top_features + [BINARY_TARGET_COLUMN]].corr().round(4)
    return {
        "target_correlations": {
            str(k): round(float(v), 4) for k, v in corr_with_target.items()
        },
        "top_features": top_features,
        "top_correlation_matrix": corr_matrix.to_dict(),
    }


def compute_duplicate_conflict_analysis(df: pd.DataFrame) -> dict[str, int]:
    """Analyze exact, non-id, and label-conflicting feature duplicates."""
    feat_cols = [c for c in PREDEFINED_FEATURE_COLUMNS if c in df.columns]
    exact_dupes = int(df.duplicated().sum())
    non_id_cols = [c for c in df.columns if c != ID_COLUMN]
    non_id_dupes = int(df.duplicated(subset=non_id_cols).sum())
    feature_dupes = int(df.duplicated(subset=feat_cols).sum())

    conflicting_groups = 0
    if BINARY_TARGET_COLUMN in df.columns and feat_cols:
        hashes = pd.util.hash_pandas_object(df[feat_cols], index=False)
        grouped_n_labels = df.groupby(hashes)[BINARY_TARGET_COLUMN].nunique()
        conflicting_groups = int((grouped_n_labels > 1).sum())

    return {
        "exact_duplicates": exact_dupes,
        "duplicates_excluding_id": non_id_dupes,
        "feature_only_duplicates": feature_dupes,
        "conflicting_feature_groups": conflicting_groups,
    }


def _plot_binary_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Generate bar + pie charts comparing Normal vs Malicious traffic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    train_counts = train_df[BINARY_TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Normal Traffic (0)", "Malicious Traffic (1)"]

    if test_df is not None and BINARY_TARGET_COLUMN in test_df.columns:
        test_counts = test_df[BINARY_TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
        x = np.arange(len(labels))
        width = 0.35
        bars1 = axes[0].bar(x - width / 2, train_counts.values, width, label="Training Set", color=COLOR_ACCENT)
        bars2 = axes[0].bar(x + width / 2, test_counts.values, width, label="Testing Set", color=COLOR_WARN)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].legend()
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            axes[0].annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
    else:
        bars = axes[0].bar(labels, train_counts.values, color=[COLOR_NORMAL, COLOR_MALICIOUS])
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=9)

    axes[0].set_title("Binary Traffic Class Distribution (Counts)")
    axes[0].set_ylabel("Number of Network Flow Records")

    axes[1].pie(
        train_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=[COLOR_NORMAL, COLOR_MALICIOUS],
        wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 1.5},
    )
    axes[1].set_title(f"Training Set Class Share (N={len(train_df):,})")

    fig.suptitle("UNSW-NB15 Binary Target Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_attack_category_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    """Generate horizontal bar chart for the 10 attack categories."""
    counts = train_df[MULTICLASS_TARGET_COLUMN].value_counts().sort_values(ascending=True)
    colors = [COLOR_NORMAL if cat == "Normal" else COLOR_MALICIOUS for cat in counts.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(counts.index, counts.values, color=colors)
    total = max(len(train_df), 1)
    for bar in bars:
        w = bar.get_width()
        pct = w / total * 100.0
        ax.annotate(
            f" {int(w):,} ({pct:.2f}%)",
            (w, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=9,
        )

    ax.set_title(f"UNSW-NB15 Attack Category Distribution (Training Set, N={len(train_df):,})")
    ax.set_xlabel("Record Count")
    ax.set_ylabel("Attack Category")
    ax.set_xlim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_categorical_by_class(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
    *,
    top_n: int = 10,
) -> None:
    """Generate stacked horizontal bar chart of top categories split by binary label."""
    top_cats = df[column].astype(str).value_counts().head(top_n).index.tolist()
    subset = df[df[column].astype(str).isin(top_cats)].copy()
    subset["_cat"] = subset[column].astype(str)
    pivot = (
        subset.groupby(["_cat", BINARY_TARGET_COLUMN])
        .size()
        .unstack(fill_value=0)
        .reindex(top_cats[::-1])
    )
    normal_vals = pivot[0] if 0 in pivot.columns else pd.Series(0, index=pivot.index)
    malicious_vals = pivot[1] if 1 in pivot.columns else pd.Series(0, index=pivot.index)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(pivot.index, normal_vals, label="Normal Traffic (0)", color=COLOR_NORMAL)
    ax.barh(
        pivot.index,
        malicious_vals,
        left=normal_vals,
        label="Malicious Traffic (1)",
        color=COLOR_MALICIOUS,
    )
    ax.set_title(title)
    ax.set_xlabel("Record Count")
    ax.set_ylabel(column)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_numerical_distributions(
    train_df: pd.DataFrame,
    output_path: Path,
    *,
    sample_size: int = 30_000,
) -> bool:
    """Plot log1p histograms for key skewed numerical flow features."""
    sampled, is_sampled = stratified_sample_df(
        train_df, max_samples=sample_size, stratify_col=BINARY_TARGET_COLUMN, random_state=RANDOM_STATE
    )
    features = [c for c in KEY_DISTRIBUTION_FEATURES if c in sampled.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes_flat[idx]
        normal_vals = np.log1p(np.clip(sampled.loc[sampled[BINARY_TARGET_COLUMN] == 0, col], 0, None))
        mal_vals = np.log1p(np.clip(sampled.loc[sampled[BINARY_TARGET_COLUMN] == 1, col], 0, None))
        ax.hist(normal_vals, bins=35, alpha=0.6, color=COLOR_NORMAL, label="Normal (0)", density=True)
        ax.hist(mal_vals, bins=35, alpha=0.6, color=COLOR_MALICIOUS, label="Malicious (1)", density=True)
        ax.set_title(f"log1p({col})")
        ax.set_xlabel(f"log1p({col})")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    sample_note = (
        f"Representative Stratified Sample (n={len(sampled):,} of {len(train_df):,})"
        if is_sampled
        else f"Full Training Set (N={len(train_df):,})"
    )
    fig.suptitle(f"Key Numerical Flow Distributions by Class — {sample_note}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return is_sampled


def _plot_outlier_boxplots(
    train_df: pd.DataFrame,
    output_path: Path,
    *,
    sample_size: int = 20_000,
) -> bool:
    """Plot class-wise box plots (on log1p scale for skewed features) to inspect outliers."""
    sampled, is_sampled = stratified_sample_df(
        train_df, max_samples=sample_size, stratify_col=BINARY_TARGET_COLUMN, random_state=RANDOM_STATE
    )
    features = [c for c in KEY_DISTRIBUTION_FEATURES if c in sampled.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    plot_df = sampled[features + [BINARY_TARGET_COLUMN]].copy()
    plot_df["Traffic Class"] = plot_df[BINARY_TARGET_COLUMN].map(
        {0: "Normal (0)", 1: "Malicious (1)"}
    )

    for idx, col in enumerate(features):
        ax = axes_flat[idx]
        plot_df[f"log1p_{col}"] = np.log1p(np.clip(plot_df[col].astype(float), 0, None))
        sns.boxplot(
            data=plot_df,
            x="Traffic Class",
            y=f"log1p_{col}",
            hue="Traffic Class",
            palette={"Normal (0)": COLOR_NORMAL, "Malicious (1)": COLOR_MALICIOUS},
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Outlier Spread: log1p({col})")
        ax.set_xlabel("")
        ax.set_ylabel(f"log1p({col})")

    sample_note = (
        f"Representative Stratified Sample (n={len(sampled):,} of {len(train_df):,})"
        if is_sampled
        else f"Full Training Set (N={len(train_df):,})"
    )
    fig.suptitle(f"Feature Outlier Box Plots by Traffic Class — {sample_note}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return is_sampled


def _plot_correlation_heatmap(
    train_df: pd.DataFrame,
    top_features: list[str],
    output_path: Path,
) -> None:
    """Plot correlation heatmap of top numerical features correlated with label."""
    cols = top_features + [BINARY_TARGET_COLUMN]
    corr_matrix = train_df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7.5},
    )
    ax.set_title(
        f"Correlation Heatmap: Top {len(top_features)} Features vs Target (Full Training Set, N={len(train_df):,})"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_eda_pipeline(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    paths: ProjectPaths | None = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Execute complete EDA on UNSW-NB15 and persist JSON summary + figures."""
    resolved_paths = ensure_directories(paths or get_paths())
    if train_df is None or test_df is None:
        loaded_train, loaded_test = load_training_and_testing_sets(paths=resolved_paths)
        train_df = loaded_train if train_df is None else train_df
        test_df = loaded_test if test_df is None else test_df

    train_profile = profile_dataset(
        train_df,
        name="UNSW_NB15_training-set",
        feature_columns=PREDEFINED_FEATURE_COLUMNS,
    )
    test_profile = profile_dataset(
        test_df,
        name="UNSW_NB15_testing-set",
        feature_columns=PREDEFINED_FEATURE_COLUMNS,
    )

    numerical_summary = compute_numerical_summary(train_df)
    categorical_summary = compute_categorical_summary(train_df)
    outlier_summary = compute_outlier_summary(train_df)
    correlation_summary = compute_correlation_analysis(train_df, top_n=15)
    duplicate_analysis = compute_duplicate_conflict_analysis(train_df)

    generated_figures: dict[str, str] = {}
    sampling_notes: dict[str, str] = {}

    if generate_plots:
        _configure_plot_style()
        fig_dir = resolved_paths.reports_figures_dir

        fig1 = fig_dir / "01_binary_class_distribution.png"
        _plot_binary_class_distribution(train_df, test_df, fig1)
        generated_figures["binary_class_distribution"] = str(fig1)

        fig2 = fig_dir / "02_attack_category_distribution.png"
        _plot_attack_category_distribution(train_df, fig2)
        generated_figures["attack_category_distribution"] = str(fig2)

        fig3 = fig_dir / "03_protocol_distribution.png"
        _plot_categorical_by_class(
            train_df,
            "proto",
            f"Top 10 Network Protocols by Traffic Class (Training Set, N={len(train_df):,})",
            fig3,
            top_n=10,
        )
        generated_figures["protocol_distribution"] = str(fig3)

        fig4 = fig_dir / "04_service_distribution.png"
        _plot_categorical_by_class(
            train_df,
            "service",
            f"Top Services by Traffic Class (Note '-' = Unspecified, N={len(train_df):,})",
            fig4,
            top_n=13,
        )
        generated_figures["service_distribution"] = str(fig4)

        fig5 = fig_dir / "05_numerical_feature_distributions.png"
        dist_sampled = _plot_numerical_distributions(train_df, fig5, sample_size=30_000)
        generated_figures["numerical_feature_distributions"] = str(fig5)
        sampling_notes["numerical_feature_distributions"] = (
            "Stratified sample of 30,000 training rows (random_state=42)."
            if dist_sampled
            else f"Full dataset ({len(train_df)} rows)."
        )

        fig6 = fig_dir / "06_outlier_boxplots.png"
        box_sampled = _plot_outlier_boxplots(train_df, fig6, sample_size=20_000)
        generated_figures["outlier_boxplots"] = str(fig6)
        sampling_notes["outlier_boxplots"] = (
            "Stratified sample of 20,000 training rows (random_state=42)."
            if box_sampled
            else f"Full dataset ({len(train_df)} rows)."
        )

        fig7 = fig_dir / "07_correlation_heatmap.png"
        _plot_correlation_heatmap(
            train_df,
            correlation_summary["top_features"],
            fig7,
        )
        generated_figures["correlation_heatmap"] = str(fig7)

        fig8 = fig_dir / "08_state_distribution.png"
        _plot_categorical_by_class(
            train_df,
            "state",
            f"Connection State Distribution by Traffic Class (Training Set, N={len(train_df):,})",
            fig8,
            top_n=9,
        )
        generated_figures["state_distribution"] = str(fig8)

    eda_summary: dict[str, Any] = {
        "analysis_scope": "Computed strictly on predefined UNSW-NB15 training set (test set summarized only for class counts).",
        "training_profile": train_profile,
        "testing_profile": test_profile,
        "duplicate_and_conflict_analysis": duplicate_analysis,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
        "outlier_summary": outlier_summary,
        "correlation_analysis": correlation_summary,
        "generated_figures": generated_figures,
        "sampling_notes": sampling_notes,
    }

    summary_path = resolved_paths.reports_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    eda_summary["summary_path"] = str(summary_path)
    return eda_summary
