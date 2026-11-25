"""Model Training and Model Evaluation pages (Pages 6 & 7)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import (
    plotly_confusion_matrix,
    plotly_pr_curves,
    plotly_roc_curves,
)
from app.components.data_access import load_json_artifact
from app.components.theme import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    render_kpi_card,
    render_page_header,
    render_setup_message,
)


def render_model_training() -> None:
    """Render Page 6: Model Training Summary & Parameters."""
    render_page_header(
        "Model Training",
        "Independent training pipelines for Logistic Regression, Decision Tree, and Random Forest",
    )

    train_sum = load_json_artifact("models/metadata/binary_training_summary.json")
    if train_sum is None:
        render_setup_message("No Training Summary Found", "python main.py train")
        return

    st.success(
        f"Binary models trained on **{train_sum['training_samples']:,}** records "
        f"using **{len(train_sum['input_features'])}** leakage-free input features."
    )

    summary_rows = []
    for key, rec in train_sum["models"].items():
        summary_rows.append(
            {
                "Model Key": key,
                "Model Name": rec["model_name"],
                "Pipeline Family": rec["model_family"],
                "Training Duration (s)": rec["training_duration_seconds"],
                "Training Samples": rec["training_samples"],
                "Trained At (UTC)": rec["trained_at_utc"],
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    for key, rec in train_sum["models"].items():
        with st.expander(f"Hyperparameters & Feature Configuration: {rec['model_name']}"):
            st.json(rec["parameters"])

    mc_meta = load_json_artifact("models/metadata/multiclass_metadata.json")
    st.subheader("Multiclass Attack Classification Extension (attack_cat)")
    if mc_meta is None:
        render_setup_message(
            "Multiclass Classifier Not Trained Yet",
            "python main.py train-multiclass",
        )
    else:
        st.info(
            f"Independent Multiclass Winner: **{mc_meta['selected_model_name']}** | "
            f"Classes ({len(mc_meta['discovered_classes'])}): `{mc_meta['discovered_classes']}`"
        )


def render_model_evaluation() -> None:
    """Render Page 7: Untouched Test-Set Evaluation & Classification Reports."""
    render_page_header(
        "Model Evaluation",
        "Untouched UNSW-NB15 testing set metrics, confusion matrices, ROC/PR curves, and latency",
    )

    eval_data = load_json_artifact("models/metadata/binary_evaluation_results.json")
    mc_meta = load_json_artifact("models/metadata/multiclass_metadata.json")

    if eval_data is None:
        render_setup_message(
            "No Binary Evaluation Results Found",
            "python main.py evaluate",
        )
        return

    models_eval = eval_data["models"]
    tab_bin, tab_mc = st.tabs(["Binary Intrusion Evaluation", "Multiclass Attack Evaluation"])

    with tab_bin:
        selected_key = st.selectbox(
            "Inspect Model Details",
            list(models_eval.keys()),
            format_func=lambda k: models_eval[k]["model_name"],
        )
        ev = models_eval[selected_key]
        m = ev["metrics"]

        r_top1, r_top2, r_top3 = st.columns(3)
        with r_top1:
            render_kpi_card("Attack Recall", f"{m['recall']*100:.2f}%", COLOR_GREEN)
        with r_top2:
            render_kpi_card("Attack Precision", f"{m['precision']*100:.2f}%", COLOR_BLUE)
        with r_top3:
            render_kpi_card("Attack F1", f"{m['f1_score']*100:.2f}%", COLOR_BLUE)

        r_bot1, r_bot2, r_bot3 = st.columns(3)
        with r_bot1:
            render_kpi_card("Accuracy", f"{m['accuracy']*100:.2f}%", COLOR_AMBER)
        with r_bot2:
            render_kpi_card("False Neg. Rate", f"{m['false_negative_rate']*100:.2f}%", COLOR_RED)
        with r_bot3:
            render_kpi_card("Latency / 1k rows", f"{m['prediction_latency_ms_per_1000']:.2f} ms", COLOR_AMBER)

        left, right = st.columns(2)
        with left:
            cm_info = ev["confusion_matrix"]
            st.plotly_chart(
                plotly_confusion_matrix(
                    cm_info["matrix"],
                    cm_info["labels"],
                    f"Confusion Matrix — {ev['model_name']}",
                ),
                width="stretch",
            )
            st.caption(
                f"TN: {cm_info['tn']:,} | FP: {cm_info['fp']:,} | "
                f"FN: {cm_info['fn']:,} | TP: {cm_info['tp']:,} | "
                f"Specificity: {m['specificity']:.4f} | FPR: {m['false_positive_rate']:.4f}"
            )
        with right:
            st.markdown("**Classification Report (Untouched Test Set)**")
            st.code(ev["classification_report_text"], language="text")

        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(plotly_roc_curves(models_eval), width="stretch")
        with r2:
            st.plotly_chart(plotly_pr_curves(models_eval), width="stretch")

    with tab_mc:
        if mc_meta is None:
            render_setup_message(
                "No Multiclass Evaluation Found",
                "python main.py train-multiclass",
            )
        else:
            mc_models = mc_meta["models"]
            mc_key = st.selectbox(
                "Select Multiclass Model",
                list(mc_models.keys()),
                index=list(mc_models.keys()).index(mc_meta["selected_model_key"]),
                format_func=lambda k: mc_models[k]["model_name"],
            )
            mc_ev = mc_models[mc_key]
            mc_m = mc_ev["metrics"]
            st.write(
                f"**Accuracy:** {mc_m['accuracy']:.4f} | "
                f"**Macro Precision:** {mc_m['macro_precision']:.4f} | "
                f"**Macro Recall:** {mc_m['macro_recall']:.4f} | "
                f"**Macro F1:** {mc_m['macro_f1']:.4f} | "
                f"**Weighted F1:** {mc_m['weighted_f1']:.4f}"
            )
            per_cls_df = pd.DataFrame.from_dict(mc_ev["per_class_metrics"], orient="index").reset_index()
            per_cls_df = per_cls_df.rename(columns={"index": "Attack Category"})
            st.dataframe(per_cls_df, width="stretch", hide_index=True)

            st.plotly_chart(
                plotly_confusion_matrix(
                    mc_ev["confusion_matrix"]["matrix"],
                    mc_ev["confusion_matrix"]["labels"],
                    f"10-Class Confusion Matrix — {mc_ev['model_name']}",
                ),
                width="stretch",
            )
