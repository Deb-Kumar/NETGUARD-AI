"""Evaluation visualization charts saved to reports/graphs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.config import ProjectPaths, ensure_directories, get_paths

MODEL_COLORS: dict[str, str] = {
    "logistic_regression": "#3B82F6",
    "decision_tree": "#F59E0B",
    "random_forest": "#10B981",
}


def plot_binary_confusion_matrices(
    evaluations: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot confusion matrices side-by-side for all evaluated binary classifiers."""
    items = list(evaluations.items())
    n_models = max(len(items), 1)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]

    for ax, (_, ev) in zip(axes, items):
        cm = np.asarray(ev["confusion_matrix"]["matrix"], dtype=int)
        row_sums = cm.sum(axis=1, keepdims=True)
        pcts = np.divide(cm, np.maximum(row_sums, 1)) * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({pcts[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred Normal", "Pred Malicious"],
            yticklabels=["Actual Normal", "Actual Malicious"],
            ax=ax,
        )
        m = ev["metrics"]
        ax.set_title(
            f"{ev['model_name']}\nRecall: {m['recall']:.3f} | FNR: {m['false_negative_rate']:.3f}",
            fontsize=10.5,
            fontweight="bold",
        )

    fig.suptitle(
        "Untouched Test Set Confusion Matrices (Positive = Malicious Traffic)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    evaluations: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot ROC curves comparing binary models on the untouched test set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, ev in evaluations.items():
        curve = ev.get("roc_curve")
        auc_val = ev["metrics"].get("roc_auc")
        if not curve or auc_val is None:
            continue
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2.2,
            color=MODEL_COLORS.get(key, "#6366F1"),
            label=f"{ev['model_name']} (AUC = {auc_val:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random Baseline (AUC = 0.5000)")
    ax.set_title("ROC Curve Comparison on UNSW-NB15 Test Set", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Attack Recall)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pr_curves(
    evaluations: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot Precision-Recall curves comparing binary models on the test set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, ev in evaluations.items():
        curve = ev.get("pr_curve")
        ap_val = ev["metrics"].get("pr_auc")
        if not curve or ap_val is None:
            continue
        ax.plot(
            curve["recall"],
            curve["precision"],
            linewidth=2.2,
            color=MODEL_COLORS.get(key, "#6366F1"),
            label=f"{ev['model_name']} (PR-AUC = {ap_val:.4f})",
        )

    ax.set_title("Precision-Recall Curve Comparison (Positive = Malicious)", fontweight="bold")
    ax.set_xlabel("Attack Recall")
    ax.set_ylabel("Attack Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison_bars(
    evaluations: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot grouped bar comparison of key security & classification metrics."""
    metric_keys = [
        ("recall", "Attack Recall"),
        ("precision", "Attack Precision"),
        ("f1_score", "Attack F1-Score"),
        ("accuracy", "Accuracy"),
        ("specificity", "Specificity"),
        ("false_negative_rate", "False Negative Rate"),
    ]
    x = np.arange(len(metric_keys))
    n_models = max(len(evaluations), 1)
    width = 0.78 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (key, ev) in enumerate(evaluations.items()):
        vals = [float(ev["metrics"][mk]) for mk, _ in metric_keys]
        offset = (idx - (n_models - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=ev["model_name"],
            color=MODEL_COLORS.get(key, "#6366F1"),
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_keys], rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison Across Cybersecurity Metrics", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multiclass_confusion_matrix(
    evaluation: dict[str, Any],
    output_path: Path,
) -> Path:
    """Plot normalized confusion matrix heatmap for the 10-class attack classifier."""
    cm = np.asarray(evaluation["confusion_matrix"]["matrix"], dtype=float)
    labels = evaluation["confusion_matrix"]["labels"]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, np.maximum(row_sums, 1.0))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(
        f"Multiclass Attack Classification Confusion Matrix (Row-Normalized) — {evaluation['model_name']}",
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Attack Category")
    ax.set_ylabel("Actual Attack Category")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_binary_evaluation_plots(
    evaluations: dict[str, dict[str, Any]],
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, str]:
    """Generate all binary evaluation comparison charts into reports/graphs/."""
    resolved_paths = ensure_directories(paths or get_paths())
    fig_dir = resolved_paths.reports_graphs_dir
    return {
        "confusion_matrices": str(
            plot_binary_confusion_matrices(
                evaluations, fig_dir / "09_confusion_matrices_binary.png"
            )
        ),
        "roc_curves": str(
            plot_roc_curves(evaluations, fig_dir / "10_roc_curves_binary.png")
        ),
        "pr_curves": str(
            plot_pr_curves(evaluations, fig_dir / "11_pr_curves_binary.png")
        ),
        "metrics_comparison": str(
            plot_model_comparison_bars(
                evaluations, fig_dir / "12_model_metrics_comparison.png"
            )
        ),
    }
