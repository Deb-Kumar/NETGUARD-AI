"""Interactive dark-navy Plotly visualizations for NETGUARD AI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.styles.theme import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_CARD,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_TEXT,
)

MODEL_PALETTE: dict[str, str] = {
    "logistic_regression": COLOR_BLUE,
    "decision_tree": COLOR_AMBER,
    "random_forest": COLOR_GREEN,
}


def _apply_dark_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=30, r=20, t=50, b=35),
        legend=dict(bgcolor="rgba(17,28,51,0.7)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def plotly_binary_donut(normal_count: int, malicious_count: int, title: str) -> go.Figure:
    """Donut chart comparing Normal vs Malicious traffic counts."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Normal Traffic (0)", "Malicious Traffic (1)"],
                values=[normal_count, malicious_count],
                hole=0.55,
                marker=dict(colors=[COLOR_GREEN, COLOR_RED]),
                textinfo="label+percent",
            )
        ]
    )
    return _apply_dark_layout(fig, title, height=340)


def plotly_attack_category_bars(cat_counts: dict[str, int], title: str) -> go.Figure:
    """Horizontal bar chart of UNSW-NB15 attack categories."""
    items = sorted(cat_counts.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLOR_GREEN if k == "Normal" else COLOR_RED for k in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,}" for v in values],
            textposition="outside",
        )
    )
    return _apply_dark_layout(fig, title, height=420)


def plotly_confusion_matrix(
    cm_matrix: list[list[int]],
    labels: list[str],
    title: str,
) -> go.Figure:
    """Interactive heatmap for binary or multiclass confusion matrices."""
    fig = px.imshow(
        cm_matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
    )
    return _apply_dark_layout(fig, title, height=400)


def plotly_roc_curves(models_eval: dict[str, dict[str, Any]]) -> go.Figure:
    """Overlay ROC curves across evaluated binary models."""
    fig = go.Figure()
    for key, ev in models_eval.items():
        curve = ev.get("roc_curve")
        auc_val = ev["metrics"].get("roc_auc")
        if not curve or auc_val is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{ev['model_name']} (AUC={auc_val:.4f})",
                line=dict(color=MODEL_PALETTE.get(key, COLOR_BLUE), width=2.5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#64748B", dash="dash"),
        )
    )
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate (Attack Recall)")
    return _apply_dark_layout(fig, "ROC Curve Comparison (Untouched Test Set)", height=420)


def plotly_pr_curves(models_eval: dict[str, dict[str, Any]]) -> go.Figure:
    """Overlay Precision-Recall curves across evaluated binary models."""
    fig = go.Figure()
    for key, ev in models_eval.items():
        curve = ev.get("pr_curve")
        ap_val = ev["metrics"].get("pr_auc")
        if not curve or ap_val is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=curve["recall"],
                y=curve["precision"],
                mode="lines",
                name=f"{ev['model_name']} (PR-AUC={ap_val:.4f})",
                line=dict(color=MODEL_PALETTE.get(key, COLOR_BLUE), width=2.5),
            )
        )
    fig.update_xaxes(title_text="Attack Recall")
    fig.update_yaxes(title_text="Attack Precision")
    return _apply_dark_layout(fig, "Precision-Recall Curve Comparison", height=420)


def plotly_model_comparison_bars(comparison_rows: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing models across cybersecurity metrics."""
    df = pd.DataFrame(comparison_rows)
    metrics = [
        "Recall (Attack)",
        "Precision (Attack)",
        "F1-Score (Attack)",
        "Accuracy",
        "Specificity",
        "FNR",
    ]
    fig = go.Figure()
    for _, row in df.iterrows():
        key = str(row["Model Key"])
        fig.add_trace(
            go.Bar(
                name=str(row["Model Name"]),
                x=metrics,
                y=[float(row[m]) for m in metrics],
                marker_color=MODEL_PALETTE.get(key, COLOR_BLUE),
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.08], title_text="Score")
    return _apply_dark_layout(fig, "Model Performance Comparison", height=420)


def plotly_cv_fold_scores(cv_models: dict[str, dict[str, Any]], metric: str = "f1") -> go.Figure:
    """Plot fold-by-fold Stratified K-Fold scores across models."""
    fig = go.Figure()
    for key, entry in cv_models.items():
        scores = entry["metrics"][metric]["fold_scores"]
        folds = [f"Fold {i + 1}" for i in range(len(scores))]
        fig.add_trace(
            go.Scatter(
                x=folds,
                y=scores,
                mode="lines+markers",
                name=f"{entry['model_name']} (Mean={entry['metrics'][metric]['mean']:.4f})",
                line=dict(color=MODEL_PALETTE.get(key, COLOR_BLUE), width=2.5),
                marker=dict(size=8),
            )
        )
    fig.update_yaxes(title_text=metric.upper())
    return _apply_dark_layout(fig, f"5-Fold Cross-Validation Scores ({metric.upper()})", height=380)
