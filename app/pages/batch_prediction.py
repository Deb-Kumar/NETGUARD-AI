"""Batch CSV Prediction page (Page 12)."""

from __future__ import annotations

import streamlit as st

from app.components.charts import plotly_binary_donut
from app.components.data_access import get_artifact_status
from app.styles.theme import render_page_header, render_setup_message
from src.database import log_prediction, sanitize_dataframe_for_csv
from src.models import MODEL_SPECS
from src.prediction import PredictionValidationError, predict_batch_csv
from src.utils.config import MAX_UPLOAD_SIZE_MB


def render_batch_prediction() -> None:
    """Render Page 12: Batch CSV Upload, Validation, Evaluation & Export."""
    render_page_header(
        "Batch CSV Prediction",
        f"Upload network flow CSV files (max {MAX_UPLOAD_SIZE_MB} MB) for validated batch classification",
    )

    status = get_artifact_status()
    if not status["best_binary_model"]:
        render_setup_message(
            "No Saved Binary Model Available",
            "python main.py train\npython main.py cross-validate\npython main.py select-best",
        )
        return

    model_options = ["best_model"] + list(MODEL_SPECS.keys())
    model_choice = st.selectbox(
        "Model for Batch Scoring",
        model_options,
        format_func=lambda k: "Best Selected Model (Default)" if k == "best_model" else MODEL_SPECS[k].display_name,
    )

    uploaded_file = st.file_uploader(
        "Upload UNSW-NB15 Feature CSV File",
        type=["csv"],
        help="Must contain the 42 UNSW-NB15 flow features (optional 'label' column enables evaluation metrics).",
    )

    if uploaded_file is None:
        st.info(
            "Upload a `.csv` file with UNSW-NB15 flow features. Prediction-only files and "
            "ground-truth files containing `label` are automatically distinguished."
        )
        return

    cache_key = f"{uploaded_file.name}:{uploaded_file.size}:{model_choice}"
    if st.button("Run Batch Classification", type="primary"):
        try:
            with st.spinner("Validating schema and scoring CSV records..."):
                batch_res = predict_batch_csv(
                    uploaded_file,
                    filename=uploaded_file.name,
                    file_size_bytes=uploaded_file.size,
                    model_key=model_choice,
                )

            dominant_class = (
                "Malicious Traffic"
                if batch_res.malicious_count >= batch_res.normal_count
                else "Normal Traffic"
            )
            mal_ratio = (
                batch_res.malicious_count / batch_res.total_records
                if batch_res.total_records
                else 0.0
            )
            log_prediction(
                prediction_type="Batch",
                model_name=batch_res.model_name,
                predicted_class=dominant_class,
                probability=round(mal_ratio, 4),
                record_reference=uploaded_file.name,
                notes=(
                    f"total={batch_res.total_records}, normal={batch_res.normal_count}, "
                    f"malicious={batch_res.malicious_count}"
                ),
            )
            st.session_state["last_batch_cache_key"] = cache_key
            st.session_state["last_batch_result"] = batch_res
        except (PredictionValidationError, FileNotFoundError) as exc:
            st.session_state.pop("last_batch_result", None)
            st.error(f"Batch CSV rejected: {exc}")

    if st.session_state.get("last_batch_cache_key") == cache_key and "last_batch_result" in st.session_state:
        batch_res = st.session_state["last_batch_result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{batch_res.total_records:,}")
        c2.metric("Predicted Normal", f"{batch_res.normal_count:,}")
        c3.metric("Predicted Malicious", f"{batch_res.malicious_count:,}")
        c4.metric("Inference Time", f"{batch_res.inference_duration_ms:.1f} ms")

        for warn in batch_res.warnings:
            st.warning(warn)

        st.plotly_chart(
            plotly_binary_donut(
                batch_res.normal_count,
                batch_res.malicious_count,
                f"Batch Prediction Distribution ({uploaded_file.name})",
            ),
            width="stretch",
        )

        if batch_res.has_ground_truth and batch_res.ground_truth_metrics is not None:
            st.subheader("Optional Evaluation Against Uploaded Ground-Truth `label`")
            gm = batch_res.ground_truth_metrics["metrics"]
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Accuracy", f"{gm['accuracy']*100:.2f}%")
            g2.metric("Attack Recall", f"{gm['recall']*100:.2f}%")
            g3.metric("Attack Precision", f"{gm['precision']*100:.2f}%")
            g4.metric("Attack F1", f"{gm['f1_score']*100:.2f}%")
            g5.metric("False Negative Rate", f"{gm['false_negative_rate']*100:.2f}%")
        else:
            st.info("Prediction-only file detected (no valid `label` column present).")

        st.subheader("Prediction Preview (First 200 Rows)")
        st.dataframe(batch_res.predictions_df.head(200), width="stretch", hide_index=True)

        safe_export_df = sanitize_dataframe_for_csv(batch_res.predictions_df)
        csv_bytes = safe_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Predictions CSV",
            data=csv_bytes,
            file_name=f"netguard_predictions_{uploaded_file.name}",
            mime="text/csv",
        )
