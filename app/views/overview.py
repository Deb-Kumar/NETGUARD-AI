"""Dashboard overview page (Page 1)."""

from __future__ import annotations

import streamlit as st

from app.components.charts import plotly_attack_category_bars, plotly_binary_donut
from app.components.data_access import get_artifact_status, load_json_artifact
from app.components.theme import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    render_kpi_card,
    render_page_header,
    render_setup_message,
)
from src.database import query_history


def render_dashboard() -> None:
    """Render Page 1: Executive Cybersecurity Overview Dashboard."""
    render_page_header(
        "NETGUARD AI — Threat Detection Command Center",
        "UNSW-NB15 Network Intrusion Detection & Attack Classification Dashboard",
    )

    status = get_artifact_status()
    model_meta = load_json_artifact("models/metadata/model_metadata.json")
    eda_summary = load_json_artifact("reports/eda_summary.json")
    mc_meta = load_json_artifact("models/metadata/multiclass_metadata.json")

    if not status["train_dataset"] or not status["test_dataset"]:
        st.error(
            "Predefined UNSW-NB15 training/testing CSV files were not detected. "
            "Check Dataset Management for placement instructions."
        )

    if model_meta is None:
        render_setup_message(
            "No Trained Binary Model Found",
            "python main.py train\npython main.py cross-validate\npython main.py tune\npython main.py select-best",
        )
    else:
        final_metrics = model_meta["final_test_evaluation"]["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_kpi_card("Selected Best Model", model_meta["selected_model_name"], COLOR_BLUE)
        with c2:
            render_kpi_card("Test Attack Recall", f"{final_metrics['recall'] * 100:.2f}%", COLOR_GREEN)
        with c3:
            render_kpi_card("Test Attack F1", f"{final_metrics['f1_score'] * 100:.2f}%", COLOR_BLUE)
        with c4:
            render_kpi_card("False Negative Rate", f"{final_metrics['false_negative_rate'] * 100:.2f}%", COLOR_RED)
        with c5:
            roc_val = final_metrics.get("roc_auc")
            render_kpi_card("ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "N/A", COLOR_AMBER)

    col_left, col_right = st.columns(2)
    with col_left:
        if eda_summary is not None:
            tr_bin = eda_summary["training_profile"]["target_distribution"]["binary_counts"]
            fig = plotly_binary_donut(
                int(tr_bin.get("0", 0)),
                int(tr_bin.get("1", 0)),
                f"Training Set Traffic Balance (N={eda_summary['training_profile']['num_records']:,})",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_setup_message("EDA Summary Missing", "python main.py run-eda")

    with col_right:
        if eda_summary is not None:
            cats = eda_summary["training_profile"]["target_distribution"]["attack_cat_counts"]
            fig = plotly_attack_category_bars(cats, "UNSW-NB15 Attack Category Breakdown")
            st.plotly_chart(fig, width="stretch")

    st.subheader("System Pipeline & Artifact Readiness")
    readiness_cols = st.columns(4)
    badges = [
        ("Dataset Validated", status["inspection_report"]),
        ("EDA Generated", status["eda_summary"]),
        ("Binary Model Ready", status["best_binary_model"]),
        ("Multiclass Model Ready", status["multiclass_model"]),
    ]
    for col, (label, ready) in zip(readiness_cols, badges):
        with col:
            cls = "ng-badge-normal" if ready else "ng-badge-warn"
            state_txt = "READY" if ready else "PENDING"
            st.markdown(
                f"<div class='ng-card'><div>{label}</div><div style='margin-top:0.4rem;'>"
                f"<span class='{cls}'>{state_txt}</span></div></div>",
                unsafe_allow_html=True,
            )

    if mc_meta is not None:
        sel_mc = mc_meta["selected_model_key"]
        mc_m = mc_meta["models"][sel_mc]["metrics"]
        st.info(
            f"**Multiclass Extension Active:** {mc_meta['selected_model_name']} | "
            f"10-Class Accuracy: {mc_m['accuracy']*100:.2f}% | "
            f"Macro F1: {mc_m['macro_f1']:.4f} | Weighted F1: {mc_m['weighted_f1']:.4f}"
        )

    st.subheader("Recent Threat Prediction Logs")
    hist = query_history(limit=8)
    if hist.empty:
        st.caption("No predictions logged yet. Run Single Traffic Prediction or Batch CSV Prediction.")
    else:
        st.dataframe(hist, width="stretch", hide_index=True)
