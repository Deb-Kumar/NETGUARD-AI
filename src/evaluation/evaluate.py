"""Model evaluation engine for binary and multiclass intrusion detection."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.utils.config import BINARY_CLASS_NAMES, ProjectPaths, ensure_directories, get_paths


def _get_positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Extract continuous positive-class (label=1) confidence scores if available."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            classes = list(getattr(model, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            return np.asarray(probs[:, pos_idx], dtype=np.float64)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores, dtype=np.float64)
    return None


def _downsample_curve(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    max_points: int = 200,
) -> tuple[list[float], list[float]]:
    """Downsample curve arrays cleanly for lightweight JSON storage and Plotly display."""
    if len(x_vals) <= max_points:
        return (
            [round(float(v), 5) for v in x_vals],
            [round(float(v), 5) for v in y_vals],
        )
    idx = np.linspace(0, len(x_vals) - 1, max_points, dtype=int)
    return (
        [round(float(x_vals[i]), 5) for i in idx],
        [round(float(y_vals[i]), 5) for i in idx],
    )


def evaluate_binary_model(
    model: Pipeline,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    model_key: str,
    model_name: str,
    split_name: str = "test",
) -> dict[str, Any]:
    """Evaluate a trained binary classifier (1 = Malicious Traffic, 0 = Normal)."""
    y_arr = np.asarray(y_true, dtype=int)
    n_samples = len(y_arr)

    start_ns = time.perf_counter_ns()
    y_pred = np.asarray(model.predict(X), dtype=int)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    accuracy = float(accuracy_score(y_arr, y_pred))
    precision = float(precision_score(y_arr, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    y_scores = _get_positive_scores(model, X)
    roc_auc: float | None = None
    pr_auc: float | None = None
    roc_points: dict[str, list[float]] | None = None
    pr_points: dict[str, list[float]] | None = None

    if y_scores is not None and len(np.unique(y_arr)) == 2:
        roc_auc = round(float(roc_auc_score(y_arr, y_scores)), 4)
        pr_auc = round(float(average_precision_score(y_arr, y_scores)), 4)
        fpr_arr, tpr_arr, _ = roc_curve(y_arr, y_scores, pos_label=1)
        prec_arr, rec_arr, _ = precision_recall_curve(y_arr, y_scores, pos_label=1)
        roc_x, roc_y = _downsample_curve(fpr_arr, tpr_arr)
        pr_x, pr_y = _downsample_curve(rec_arr, prec_arr)
        roc_points = {"fpr": roc_x, "tpr": roc_y}
        pr_points = {"recall": pr_x, "precision": pr_y}

    target_names = [BINARY_CLASS_NAMES[0], BINARY_CLASS_NAMES[1]]
    report_dict = classification_report(
        y_arr,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_arr,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0,
    )

    return {
        "model_key": model_key,
        "model_name": model_name,
        "split_name": split_name,
        "num_samples": n_samples,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "false_positive_rate": round(fpr, 4),
            "specificity": round(specificity, 4),
            "false_negative_rate": round(fnr, 4),
            "prediction_latency_ms_total": round(elapsed_ms, 2),
            "prediction_latency_ms_per_1000": round(
                elapsed_ms / max(n_samples, 1) * 1000.0, 3
            ),
        },
        "confusion_matrix": {
            "labels": target_names,
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }


def evaluate_multiclass_model(
    model: Pipeline,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    model_key: str,
    model_name: str,
    split_name: str = "test",
) -> dict[str, Any]:
    """Evaluate an independent multiclass attack-category classifier."""
    y_arr = np.asarray(y_true, dtype=str)
    n_samples = len(y_arr)
    labels = sorted(set(y_arr.tolist()) | set(map(str, getattr(model, "classes_", []))))

    start_ns = time.perf_counter_ns()
    y_pred = np.asarray(model.predict(X), dtype=str)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    cm = confusion_matrix(y_arr, y_pred, labels=labels)
    report_dict = classification_report(
        y_arr,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_arr,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for cls in labels:
        if cls in report_dict:
            per_class[cls] = {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1_score": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"]),
            }

    return {
        "model_key": model_key,
        "model_name": model_name,
        "split_name": split_name,
        "num_samples": n_samples,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_arr, y_pred)), 4),
            "macro_precision": round(
                float(precision_score(y_arr, y_pred, average="macro", zero_division=0)),
                4,
            ),
            "macro_recall": round(
                float(recall_score(y_arr, y_pred, average="macro", zero_division=0)),
                4,
            ),
            "macro_f1": round(
                float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
                4,
            ),
            "weighted_f1": round(
                float(f1_score(y_arr, y_pred, average="weighted", zero_division=0)),
                4,
            ),
            "prediction_latency_ms_total": round(elapsed_ms, 2),
            "prediction_latency_ms_per_1000": round(
                elapsed_ms / max(n_samples, 1) * 1000.0, 3
            ),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def build_comparison_table(evaluations: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Assemble a tidy DataFrame comparing evaluated binary models."""
    rows: list[dict[str, Any]] = []
    for ev in evaluations:
        m = ev["metrics"]
        cm = ev["confusion_matrix"]
        rows.append(
            {
                "Model Key": ev["model_key"],
                "Model Name": ev["model_name"],
                "Accuracy": m["accuracy"],
                "Precision (Attack)": m["precision"],
                "Recall (Attack)": m["recall"],
                "F1-Score (Attack)": m["f1_score"],
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "Specificity": m["specificity"],
                "FPR": m["false_positive_rate"],
                "FNR": m["false_negative_rate"],
                "Latency (ms/1k)": m["prediction_latency_ms_per_1000"],
                "False Negatives": cm["fn"],
                "False Positives": cm["fp"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Recall (Attack)", "F1-Score (Attack)"], ascending=False)
    return df.reset_index(drop=True)


def save_evaluation_results(
    evaluations: dict[str, dict[str, Any]],
    *,
    filename: str = "binary_evaluation_results.json",
    paths: ProjectPaths | None = None,
) -> Path:
    """Persist structured evaluation metrics and comparison table to models/metadata/."""
    resolved_paths = ensure_directories(paths or get_paths())
    comparison_df = build_comparison_table(list(evaluations.values()))
    payload = {
        "models": evaluations,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }
    out_path = resolved_paths.model_metadata_dir / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
