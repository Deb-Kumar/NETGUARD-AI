"""Single Traffic Prediction page (Page 11)."""

from __future__ import annotations

import streamlit as st

from app.components.data_access import (
    get_artifact_status,
    get_categorical_value_choices,
    load_feature_descriptions_map,
    load_json_artifact,
    load_sample_split,
)
from app.styles.theme import render_page_header, render_setup_message
from src.data.validator import PREDEFINED_FEATURE_COLUMNS
from src.database import log_prediction
from src.models import MODEL_SPECS
from src.prediction import (
    PredictionValidationError,
    get_expected_feature_schema,
    predict_single_record,
)
from src.utils.config import get_paths

INTEGER_FEATURE_HINT: set[str] = {
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
}


def _get_median_baselines() -> dict[str, float]:
    """Read training median statistics from EDA summary without fabricating attack inputs."""
    eda = load_json_artifact("reports/experiments/eda_summary.json")
    if not eda or "numerical_summary" not in eda:
        return {}
    return {
        col: float(stats.get("median", 0.0))
        for col, stats in eda["numerical_summary"].items()
    }


def render_single_prediction() -> None:
    """Render Page 11: Dynamic Single-Record Network Traffic Prediction."""
    render_page_header(
        "Single Traffic Prediction",
        "Classify an individual network flow record using the saved preprocessing & model pipeline",
    )

    status = get_artifact_status()
    if not status["best_binary_model"]:
        render_setup_message(
            "No Saved Binary Model Available",
            "python main.py train\npython main.py cross-validate\npython main.py select-best",
        )
        return

    paths = get_paths()
    active_model_key = st.session_state.get("active_binary_model_key", "best_model")
    available_models = ["best_model"] + [
        k for k in MODEL_SPECS if (paths.binary_models_dir / f"{k}.joblib").is_file()
    ]
    model_choice = st.selectbox(
        "Prediction Model",
        available_models,
        index=available_models.index(active_model_key) if active_model_key in available_models else 0,
        format_func=lambda k: "Best Selected Model (Default)" if k == "best_model" else MODEL_SPECS[k].display_name,
    )

    schema = get_expected_feature_schema()
    cat_choices = get_categorical_value_choices()
    medians = _get_median_baselines()
    descriptions = load_feature_descriptions_map()

    st.markdown("#### Input Initialization")
    init_col1, init_col2 = st.columns([1.3, 1.7])
    with init_col1:
        init_mode = st.radio(
            "Form Input Baseline",
            ["Training Median Baseline (Neutral Reference)", "Inspect Actual Test Set Record"],
            horizontal=False,
        )

    prefill_row: dict[str, object] = {}
    reference_label_info: str | None = None
    record_ref = "manual-single-flow"

    if init_mode == "Inspect Actual Test Set Record":
        test_sample = load_sample_split("test", nrows=500)
        if test_sample is not None and not test_sample.empty:
            with init_col2:
                filter_cls = st.selectbox(
                    "Filter Reference Test Rows",
                    ["All", "Normal (0)", "Malicious (1)"],
                )
                subset = test_sample
                if filter_cls == "Normal (0)":
                    subset = test_sample[test_sample["label"] == 0]
                elif filter_cls == "Malicious (1)":
                    subset = test_sample[test_sample["label"] == 1]

                if not subset.empty:
                    row_idx = st.selectbox(
                        "Choose Actual UNSW-NB15 Testing Row (by id)",
                        subset["id"].tolist(),
                        format_func=lambda rid: f"Test Row id={rid}",
                    )
                    matched = subset[subset["id"] == row_idx].iloc[0]
                    prefill_row = matched[list(PREDEFINED_FEATURE_COLUMNS)].to_dict()
                    reference_label_info = (
                        f"Actual Test Ground Truth for `id={row_idx}`: "
                        f"label=`{matched['label']}` | attack_cat=`{matched['attack_cat']}`"
                    )
                    record_ref = f"test-id-{row_idx}"
                else:
                    st.warning("No matching rows in sample.")
        else:
            st.warning("Test dataset sample unavailable; using training median baseline.")
    else:
        with init_col2:
            st.caption(
                "Fields initialize to training-set medians solely as neutral numeric placeholders "
                "(not an attack signature). Adjust any fields before running prediction."
            )

    if reference_label_info:
        st.info(reference_label_info)

    form_values: dict[str, object] = {}
    with st.form("single_prediction_form"):
        tab_core, tab_vol, tab_rate, tab_conn = st.tabs(
            [
                "Protocol & State (3)",
                "Packets & Bytes (12)",
                "Rates, TTL & TCP Timing (14)",
                "Connection Tracking (13)",
            ]
        )

        core_cols = ["proto", "service", "state", "dur"]
        vol_cols = [
            "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
            "smean", "dmean", "trans_depth", "response_body_len", "swin", "dwin",
        ]
        rate_cols = [
            "rate", "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt",
            "sjit", "djit", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
        ]
        conn_cols = [
            col for col in schema["feature_names"]
            if col not in set(core_cols + vol_cols + rate_cols)
        ]

        def _render_fields(cols_list: list[str]) -> None:
            grid = st.columns(3)
            for idx, col_name in enumerate(cols_list):
                help_txt = descriptions.get(col_name, {}).get("description", "")
                with grid[idx % 3]:
                    if col_name in schema["categorical_columns"]:
                        opts = list(cat_choices.get(col_name, ["-"]))
                        default_val = str(prefill_row.get(col_name, opts[0]))
                        if default_val not in opts:
                            opts = [default_val] + opts
                        form_values[col_name] = st.selectbox(
                            col_name,
                            options=opts,
                            index=opts.index(default_val),
                            key=f"single_{init_mode}_{record_ref}_{col_name}",
                            help=help_txt,
                        )
                    else:
                        default_num = float(prefill_row.get(col_name, medians.get(col_name, 0.0)))
                        if col_name in INTEGER_FEATURE_HINT:
                            form_values[col_name] = st.number_input(
                                col_name,
                                min_value=0,
                                value=max(0, int(round(default_num))),
                                step=1,
                                key=f"single_{init_mode}_{record_ref}_{col_name}",
                                help=help_txt,
                            )
                        else:
                            form_values[col_name] = st.number_input(
                                col_name,
                                min_value=0.0,
                                value=max(0.0, float(default_num)),
                                format="%.6f",
                                key=f"single_{init_mode}_{record_ref}_{col_name}",
                                help=help_txt,
                            )

        with tab_core:
            _render_fields(core_cols)
        with tab_vol:
            _render_fields(vol_cols)
        with tab_rate:
            _render_fields(rate_cols)
        with tab_conn:
            _render_fields(conn_cols)

        submitted = st.form_submit_button("Analyze Network Flow", type="primary")

    if submitted:
        try:
            result = predict_single_record(
                form_values,
                model_key=model_choice,
                include_multiclass=True,
            )
            log_prediction(
                prediction_type="Single",
                model_name=result.model_name,
                predicted_class=result.predicted_class,
                probability=result.malicious_probability,
                record_reference=record_ref,
                notes=(
                    f"multiclass={result.multiclass_attack_category}"
                    if result.multiclass_attack_category
                    else ""
                ),
            )

            badge_cls = "ng-badge-malicious" if result.predicted_label == 1 else "ng-badge-normal"
            st.markdown(
                f"### Prediction Result: <span class='{badge_cls}'>{result.predicted_class.upper()}</span>",
                unsafe_allow_html=True,
            )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Model Used", result.model_name)
            r2.metric(
                "Malicious Probability",
                f"{result.malicious_probability * 100:.2f}%" if result.malicious_probability is not None else "N/A",
            )
            r3.metric(
                "Normal Probability",
                f"{result.normal_probability * 100:.2f}%" if result.normal_probability is not None else "N/A",
            )
            r4.metric(
                "Multiclass Category",
                result.multiclass_attack_category or "Not Trained",
            )
            st.caption(f"Timestamp (UTC): {result.timestamp_utc} | Logged to SQLite prediction history.")

            for warn in result.warnings:
                st.warning(warn)

        except (PredictionValidationError, FileNotFoundError) as exc:
            st.error(f"Prediction error: {exc}")
