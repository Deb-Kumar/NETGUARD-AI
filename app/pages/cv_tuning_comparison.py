"""Cross-Validation, Hyperparameter Tuning, and Model Comparison pages (Pages 8-10)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import plotly_cv_fold_scores, plotly_model_comparison_bars
from app.components.data_access import load_json_artifact
from app.styles.theme import (
    COLOR_BLUE,
    COLOR_GREEN,
    render_kpi_card,
    render_page_header,
    render_setup_message,
)


def render_cross_validation() -> None:
    """Render Page 8: Stratified 5-Fold Cross-Validation Results."""
    render_page_header(
        "Cross-Validation",
        "Stratified K-Fold cross-validation computed strictly on training data",
    )

    cv_data = load_json_artifact("models/metadata/cv_results.json")
    if cv_data is None:
        render_setup_message("Cross-Validation Results Not Found", "python main.py cross-validate")
        return

    st.info(
        f"**Folds:** {cv_data['n_splits']} | **Random State:** {cv_data['random_state']} | "
        f"**Scope:** {cv_data['dataset_scope']}"
    )

    st.subheader("Cross-Validation Comparison Table")
    st.dataframe(pd.DataFrame(cv_data["comparison_table"]), width="stretch", hide_index=True)

    metric_choice = st.selectbox("Select Metric for Fold Breakdown", ["recall", "f1", "precision", "accuracy", "roc_auc"], index=1)
    st.plotly_chart(plotly_cv_fold_scores(cv_data["models"], metric=metric_choice), width="stretch")

    st.subheader("Per-Fold Scores by Model")
    for _, entry in cv_data["models"].items():
        with st.expander(f"{entry['model_name']} — Fold Details ({entry['duration_seconds']}s)"):
            fold_rows = []
            for m_name, m_vals in entry["metrics"].items():
                fold_rows.append(
                    {
                        "Metric": m_name.upper(),
                        "Mean": m_vals["mean"],
                        "Std Dev": m_vals["std"],
                        **{f"Fold {i+1}": s for i, s in enumerate(m_vals["fold_scores"])},
                    }
                )
            st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)


def render_hyperparameter_tuning() -> None:
    """Render Page 9: GridSearchCV Hyperparameter Tuning Results."""
    render_page_header(
        "Hyperparameter Tuning",
        "GridSearchCV search spaces, best parameters, and validation performance (training data only)",
    )

    tuning = load_json_artifact("models/metadata/tuning_results.json")
    if tuning is None:
        render_setup_message("Tuning Results Not Found", "python main.py tune")
        return

    st.info(
        f"**CV Folds:** {tuning['cv_folds']} | **Primary Refit Metric:** `{tuning['primary_scoring']}` | "
        f"**Search Scope:** {tuning['search_scope']}"
    )

    summary_rows = []
    for _, entry in tuning["models"].items():
        m = entry["best_cv_metrics"]
        summary_rows.append(
            {
                "Model": entry["model_name"],
                "Candidates": entry["num_candidates"],
                f"Best CV ({tuning['primary_scoring']})": entry["best_cv_score"],
                "CV Recall": m["recall"],
                "CV Precision": m["precision"],
                "CV F1": m["f1"],
                "CV ROC-AUC": m["roc_auc"],
                "Search Time (s)": entry["tuning_duration_seconds"],
                "Full Refit (s)": entry["full_refit_duration_seconds"],
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    for _, entry in tuning["models"].items():
        st.markdown(f"### {entry['model_name']}")
        st.write("**Best Parameters:**")
        st.json(entry["best_params"])
        with st.expander(f"Top GridSearchCV Candidates — {entry['model_name']}"):
            cand_rows = []
            for cand in entry["top_candidates"]:
                cand_rows.append(
                    {
                        "Rank": cand["rank"],
                        "Params": str(cand["params"]),
                        "CV Recall": cand.get("cv_recall"),
                        "CV Precision": cand.get("cv_precision"),
                        "CV F1": cand.get("cv_f1"),
                        "CV ROC-AUC": cand.get("cv_roc_auc"),
                        "Mean Fit (s)": cand.get("mean_fit_time_s"),
                    }
                )
            st.dataframe(pd.DataFrame(cand_rows), width="stretch", hide_index=True)


def render_model_comparison() -> None:
    """Render Page 10: Model Comparison & Best Model Selection."""
    render_page_header(
        "Model Comparison & Selection",
        "Validation-driven model selection distinguishing CV selection metrics from final untouched test metrics",
    )

    meta = load_json_artifact("models/metadata/model_metadata.json")
    if meta is None:
        render_setup_message("Model Selection Metadata Not Found", "python main.py select-best")
        return

    c1, c2 = st.columns(2)
    with c1:
        render_kpi_card("Selected Best Model", meta["selected_model_name"], COLOR_GREEN)
    with c2:
        render_kpi_card("Selection Strategy", meta["selection_strategy"], COLOR_BLUE)

    st.caption(meta["selection_strategy_description"])

    st.subheader("1. Validation Ranking (Used for Model Selection — No Test Data)")
    val_rows = []
    for row in meta["validation_ranking"]:
        vm = row["validation_metrics"]
        val_rows.append(
            {
                "Model": row["model_name"],
                "Selection Score": row["selection_score"],
                "Val Recall": vm["recall"],
                "Val F1": vm["f1"],
                "Val Precision": vm["precision"],
                "Val FNR": row["validation_fnr"],
                "Validation Source": row["validation_source"],
            }
        )
    st.dataframe(pd.DataFrame(val_rows), width="stretch", hide_index=True)

    st.subheader("2. Final Untouched Test Set Comparison")
    comp_rows = meta["all_models_test_comparison"]
    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
    st.plotly_chart(plotly_model_comparison_bars(comp_rows), width="stretch")

    with st.expander("Selected Model Saved Parameters & Artifact Metadata"):
        st.json(
            {
                "selected_model_key": meta["selected_model_key"],
                "training_date_utc": meta["training_date_utc"],
                "target_name": meta["target_name"],
                "feature_count": len(meta["feature_names"]),
                "model_parameters": meta["model_parameters"],
            }
        )
