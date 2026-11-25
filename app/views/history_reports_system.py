"""Prediction History, Reports & Exports, About Project, and Settings (Pages 13-16)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.data_access import get_artifact_status, list_saved_figures, load_json_artifact
from app.components.theme import render_page_header
from src.database import clear_history, export_history_csv, query_history
from src.models import MODEL_SPECS
from src.utils.config import (
    APP_VERSION,
    CV_FOLDS,
    DEFAULT_N_JOBS,
    MAX_UPLOAD_SIZE_MB,
    PROJECT_TITLE,
    RANDOM_STATE,
    get_paths,
)


def render_prediction_history() -> None:
    """Render Page 13: SQLite Prediction History with Search, Filtering & Export."""
    render_page_header(
        "Prediction History",
        "Audit log stored in SQLite (no sensitive raw packet/IP payloads retained)",
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        search = st.text_input("Search reference / model / notes", "")
    with f2:
        pred_type = st.selectbox("Prediction Type", ["All", "Single", "Batch"])
    with f3:
        pred_class = st.selectbox("Predicted Class", ["All", "Normal Traffic", "Malicious Traffic"])
    with f4:
        limit = st.number_input("Max Rows", min_value=10, max_value=2000, value=250, step=50)

    d1, d2 = st.columns(2)
    with d1:
        use_date_filter = st.checkbox("Filter by UTC Date Range", value=False)
    start_date_str = None
    end_date_str = None
    if use_date_filter:
        with d2:
            c_start, c_end = st.columns(2)
            start_d = c_start.date_input("Start Date")
            end_d = c_end.date_input("End Date")
            start_date_str = start_d.isoformat()
            end_date_str = end_d.isoformat()

    hist_df = query_history(
        search_text=search or None,
        prediction_type=pred_type,
        predicted_class=pred_class,
        start_date=start_date_str,
        end_date=end_date_str,
        limit=int(limit),
    )

    st.write(f"**Matching Records:** {len(hist_df):,}")
    if hist_df.empty:
        st.info("No prediction history records match the selected filters.")
    else:
        st.dataframe(hist_df, width="stretch", hide_index=True)

        csv_payload = export_history_csv(
            search_text=search or None,
            prediction_type=pred_type,
            predicted_class=pred_class,
            start_date=start_date_str,
            end_date=end_date_str,
            limit=int(limit),
        )
        st.download_button(
            "Export Filtered History CSV",
            data=str(csv_payload).encode("utf-8"),
            file_name="netguard_prediction_history.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Clear Prediction History")
    confirm_clear = st.checkbox("I confirm permanently deleting all stored SQLite prediction history records.")
    if st.button("Clear All History", disabled=not confirm_clear, type="secondary"):
        deleted = clear_history(confirm=True)
        st.success(f"Deleted {deleted} prediction history record(s).")
        st.rerun()


def render_reports_exports() -> None:
    """Render Page 14: Downloadable Reports, Evaluation Tables & Figures Gallery."""
    render_page_header(
        "Reports and Exports",
        "Download generated JSON/CSV evaluation reports and inspect publication-ready figures",
    )

    paths = get_paths()
    report_files = [
        ("Dataset Inspection Report", paths.reports_dir / "dataset_inspection_report.json"),
        ("EDA Summary Report", paths.reports_dir / "eda_summary.json"),
        ("Preprocessing Configuration", paths.model_metadata_dir / "preprocessing_config.json"),
        ("Binary Training Summary", paths.model_metadata_dir / "binary_training_summary.json"),
        ("5-Fold Cross-Validation Results", paths.model_metadata_dir / "cv_results.json"),
        ("GridSearchCV Tuning Results", paths.model_metadata_dir / "tuning_results.json"),
        ("Binary Evaluation Results", paths.model_metadata_dir / "binary_evaluation_results.json"),
        ("Final Best Model Metadata", paths.model_metadata_dir / "model_metadata.json"),
        ("Multiclass Metadata", paths.model_metadata_dir / "multiclass_metadata.json"),
        ("MCA Final-Year PBL Report", paths.reports_dir / "MCA_PBL_PROJECT_REPORT.md"),
    ]

    st.subheader("Downloadable Artifacts & Technical Reports")
    for label, fpath in report_files:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(f"**{label}** (`{fpath.name}`)")
        with col_b:
            if fpath.is_file():
                mime_type = "text/markdown" if fpath.suffix.lower() == ".md" else "application/json"
                st.download_button(
                    f"Download {fpath.suffix}",
                    data=fpath.read_bytes(),
                    file_name=fpath.name,
                    mime=mime_type,
                    key=f"dl_{fpath.name}",
                )
            else:
                st.caption("Not generated")

    eval_data = load_json_artifact("models/metadata/binary_evaluation_results.json")
    if eval_data and eval_data.get("comparison_table"):
        import pandas as pd

        comp_csv = pd.DataFrame(eval_data["comparison_table"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Model Comparison Table (CSV)",
            data=comp_csv,
            file_name="netguard_model_comparison.csv",
            mime="text/csv",
        )

    st.subheader("Generated Visualization Gallery")
    figures = list_saved_figures()
    if not figures:
        st.info("No PNG charts found in reports/figures/. Run EDA or model evaluation first.")
    else:
        for title, fig_path in figures:
            with st.expander(title):
                st.image(str(fig_path), caption=fig_path.name, width="stretch")


def render_about_project() -> None:
    """Render Page 15: Project Architecture, Methodology & Limitations."""
    render_page_header(
        "About NETGUARD AI",
        PROJECT_TITLE,
    )

    st.markdown(
        f"""
        ### Overview
        **NETGUARD AI** (`v{APP_VERSION}`) is an MCA Final-Year PBL cybersecurity machine learning system
        developed in Python 3.12 using **Scikit-learn**, **Pandas**, **Plotly**, **SQLite**, and **Streamlit**.

        ### Machine Learning Methodology
        - **Dataset:** UNSW-NB15 predefined training (`175,341` records) and testing (`82,332` records) partitions.
        - **Leakage Safeguards:** Excludes `id`, `attack_cat`, and `label` from binary input features; fits all
          imputers, scalers, and `OneHotEncoder(handle_unknown="ignore")` strictly inside training folds.
        - **Algorithms:** Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
        - **Validation & Tuning:** Stratified 5-Fold Cross-Validation + GridSearchCV on training data only.
        - **Selection Criterion:** Composite cybersecurity metric prioritizing **Attack Recall**, **F1-Score**,
          and **False Negative Rate** rather than raw accuracy alone.
        - **Multiclass Extension:** Independent 10-class `attack_cat` pipeline trained without binary `label` leakage.

        ### Operational Limitations & Ethics Notice
        - Designed for offline flow-record analysis (single record or CSV export), **not** live packet capture.
        - Automated ML predictions provide probabilistic decision support and do **not** guarantee the presence
          or absence of an actual network intrusion.

        ### Dataset References
        1. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*. IEEE MilCIS.
        2. Moustafa, N., & Slay, J. (2016). *The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set*. Information Security Journal.
        """
    )


def render_settings() -> None:
    """Render Page 16: System Settings, Active Model Selection & Health Check."""
    render_page_header(
        "Settings",
        "Configure dashboard prediction defaults and inspect system artifact status",
    )

    st.subheader("Inference Model Preference (Session)")
    options = ["best_model"] + list(MODEL_SPECS.keys())
    current = st.session_state.get("active_binary_model_key", "best_model")
    chosen = st.selectbox(
        "Default Binary Model for Single/Batch Prediction",
        options,
        index=options.index(current) if current in options else 0,
        format_func=lambda k: "Best Selected Model (Auto)" if k == "best_model" else MODEL_SPECS[k].display_name,
    )
    if st.button("Save Session Model Preference"):
        st.session_state["active_binary_model_key"] = chosen
        st.success(f"Updated default session model to `{chosen}`.")

    st.subheader("Runtime & Security Configuration")
    st.json(
        {
            "app_version": APP_VERSION,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "default_n_jobs": DEFAULT_N_JOBS,
            "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
            "trusted_model_format": ".joblib inside models/",
        }
    )

    st.subheader("Artifact Health Status")
    status = get_artifact_status()
    rows = [{"Component": k, "Ready": "Yes" if v else "No"} for k, v in status.items()]
    import pandas as pd

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if st.button("Clear Streamlit Data Cache"):
        st.cache_data.clear()
        st.success("Cleared cached metadata and dataset previews.")
