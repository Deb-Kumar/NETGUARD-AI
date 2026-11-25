"""EDA & Analytics and Data Preprocessing pages (Pages 4 & 5)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.charts import plotly_attack_category_bars, plotly_binary_donut
from app.components.data_access import list_saved_figures, load_json_artifact
from app.components.theme import COLOR_CARD, COLOR_TEXT, render_page_header, render_setup_message


def render_eda_analytics() -> None:
    """Render Page 4: Exploratory Data Analysis & Stored Visualizations."""
    render_page_header(
        "EDA and Analytics",
        "Statistical distributions, outlier bounds, protocol/service breakdown, and correlations",
    )

    eda = load_json_artifact("reports/eda_summary.json")
    if eda is None:
        render_setup_message("EDA Results Not Generated", "python main.py run-eda")
        return

    st.info(eda["analysis_scope"])
    tr = eda["training_profile"]
    dupes = eda["duplicate_and_conflict_analysis"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Training Rows", f"{tr['num_records']:,}")
    k2.metric("Service '-' Placeholders", f"{tr['dash_placeholders'].get('service', 0):,}")
    k3.metric("Feature-Only Duplicates", f"{dupes['feature_only_duplicates']:,}")
    k4.metric("Conflicting Feature Groups", f"{dupes['conflicting_feature_groups']:,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions & Correlations", "Numerical & Outliers", "Protocol & Service", "Generated PNG Charts"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            bin_cnt = tr["target_distribution"]["binary_counts"]
            st.plotly_chart(
                plotly_binary_donut(int(bin_cnt["0"]), int(bin_cnt["1"]), "Binary Class Balance"),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                plotly_attack_category_bars(
                    tr["target_distribution"]["attack_cat_counts"],
                    "Attack Category Counts",
                ),
                width="stretch",
            )

        st.subheader("Top Features Correlated with Malicious Label")
        corr_items = list(eda["correlation_analysis"]["target_correlations"].items())[:15]
        corr_df = pd.DataFrame(corr_items, columns=["Feature", "Pearson Correlation with Label"])
        fig_corr = px.bar(
            corr_df,
            x="Pearson Correlation with Label",
            y="Feature",
            orientation="h",
            color="Pearson Correlation with Label",
            color_continuous_scale="RdBu_r",
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLOR_CARD,
            font=dict(color=COLOR_TEXT),
            height=450,
        )
        st.plotly_chart(fig_corr, width="stretch")

    with tab2:
        st.subheader("Numerical Feature Summary (Training Set)")
        num_df = pd.DataFrame.from_dict(eda["numerical_summary"], orient="index").reset_index()
        num_df = num_df.rename(columns={"index": "Feature"})
        st.dataframe(num_df, width="stretch", hide_index=True)

        st.subheader("IQR Outlier Analysis")
        out_df = pd.DataFrame.from_dict(eda["outlier_summary"], orient="index").reset_index()
        out_df = out_df.rename(columns={"index": "Feature"})
        st.dataframe(out_df, width="stretch", hide_index=True)

    with tab3:
        for cat_col in ("proto", "service", "state"):
            info = eda["categorical_summary"][cat_col]
            st.markdown(f"#### `{cat_col}` (Unique Values: {info['unique_count']})")
            rows = []
            for k, cnt in info["top_10_counts"].items():
                rows.append(
                    {
                        "Category": k,
                        "Count": cnt,
                        "Attack Rate (%)": info["top_10_attack_rate_pct"].get(k, 0.0),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab4:
        sampling_notes = eda.get("sampling_notes", {})
        if sampling_notes:
            st.caption(f"Sampling Disclosures: {sampling_notes}")
        for title, fig_path in list_saved_figures():
            if any(fig_path.name.startswith(prefix) for prefix in ("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_")):
                st.image(str(fig_path), caption=title, width="stretch")


def render_data_preprocessing() -> None:
    """Render Page 5: Preprocessing Pipelines, Leakage Prevention & Imbalance Study."""
    render_page_header(
        "Data Preprocessing",
        "Leakage-free Scikit-learn ColumnTransformer pipelines, feature engineering, and class imbalance analysis",
    )

    prep = load_json_artifact("models/metadata/preprocessing_config.json")
    train_sum = load_json_artifact("models/metadata/binary_training_summary.json")

    if prep is None:
        render_setup_message("Preprocessing Configuration Not Found", "python main.py train")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Leakage Safeguards")
        st.write(f"**Excluded Columns:** `{prep['excluded_columns']}`")
        st.write("**Fit Scope:** Fitted strictly on training split folds; never touched by test data.")
        st.write(f"**Feature Engineering Enabled:** `{prep['feature_engineering_enabled']}`")
        st.write(f"**Engineered Features:** `{prep['engineered_features']}`")
    with c2:
        st.subheader("Column Separation")
        st.metric("Numerical Features (incl. engineered)", len(prep["numerical_columns"]))
        st.metric("Categorical Features", len(prep["categorical_columns"]))
        st.write(f"Categorical: `{prep['categorical_columns']}`")

    st.subheader("Estimator-Specific Pipelines")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Linear Pipeline (Logistic Regression)**")
        st.json(prep["pipelines"]["linear"])
    with p2:
        st.markdown("**Tree Pipeline (Decision Tree / Random Forest)**")
        st.json(prep["pipelines"]["tree"])

    st.subheader("Class Imbalance Investigation & SMOTE Policy")
    imb = prep["class_imbalance_analysis"]
    st.json(imb)

    if train_sum and train_sum.get("class_weight_study"):
        st.subheader("Empirical `class_weight='balanced'` Effect (Training CV)")
        study = train_sum["class_weight_study"]
        st.caption(f"Scope: {study['dataset_scope']} ({study['cv_folds']}-fold Stratified CV)")
        rows = []
        for model_key, variants in study["results"].items():
            for weight_mode, metrics in variants.items():
                rows.append({"Model": model_key, "class_weight": weight_mode, **metrics})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
