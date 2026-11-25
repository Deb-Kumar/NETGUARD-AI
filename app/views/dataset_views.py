"""Dataset Management and Dataset Explorer pages (Pages 2 & 3)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.data_access import (
    load_feature_descriptions_map,
    load_json_artifact,
    load_sample_split,
)
from app.components.theme import render_page_header, render_setup_message
from src.utils.config import get_paths


def render_dataset_management() -> None:
    """Render Page 2: Dataset Inventory, Schema Validation & Feature Dictionary."""
    render_page_header(
        "Dataset Management",
        "UNSW-NB15 dataset file inventory, schema validation, and official feature dictionary",
    )

    paths = get_paths()
    inventory_rows: list[dict[str, str]] = []
    checks = [
        ("Training Set (Predefined)", paths.train_dataset_path),
        ("Testing Set (Predefined)", paths.test_dataset_path),
        ("Feature Descriptions", paths.features_description_path),
        ("Ground Truth Table", paths.ground_truth_path),
        ("List of Events", paths.list_events_path),
    ]
    for name, path in paths.raw_part_paths.items():
        checks.append((f"Raw Split ({name})", path))

    for label, p in checks:
        exists = p is not None and p.is_file()
        size_mb = f"{p.stat().st_size / (1024 * 1024):.2f} MB" if exists and p is not None else "—"
        inventory_rows.append(
            {
                "Dataset Role": label,
                "Filename": p.name if p is not None else "Not Found",
                "Status": "Available" if exists else "Missing",
                "Size": size_mb,
            }
        )
    st.subheader("Detected UNSW-NB15 Files")
    st.dataframe(pd.DataFrame(inventory_rows), width="stretch", hide_index=True)

    inspection = load_json_artifact("reports/dataset_inspection_report.json")
    if inspection is None:
        render_setup_message(
            "Dataset Inspection Report Not Found",
            "python main.py inspect-data --save-report",
        )
    else:
        st.subheader("Schema & Integrity Validation")
        val = inspection["validation"]
        if val["is_valid"]:
            st.success("Predefined training and testing datasets passed schema & target consistency validation.")
        else:
            st.error(f"Validation failed: {val['errors']}")

        for warn in val.get("warnings", []):
            st.warning(warn)

        c1, c2 = st.columns(2)
        with c1:
            tr = inspection["training_set"]
            st.metric("Training Records", f"{tr['num_records']:,}")
            st.metric("Training Columns", tr["num_columns"])
            st.metric("Training Missing Cells", tr["missing_values"]["total_missing_cells"])
        with c2:
            te = inspection["testing_set"]
            st.metric("Testing Records", f"{te['num_records']:,}")
            st.metric("Testing Columns", te["num_columns"])
            st.metric("Testing Missing Cells", te["missing_values"]["total_missing_cells"])

    st.subheader("Official UNSW-NB15 Feature Dictionary")
    feat_map = load_feature_descriptions_map()
    if feat_map:
        query = st.text_input("Filter feature dictionary by name or description", "")
        rows = [
            {"Column": col, "Type": info["type"], "Description": info["description"]}
            for col, info in sorted(feat_map.items())
            if not query or query.lower() in col.lower() or query.lower() in info["description"].lower()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("NUSW-NB15_features.csv could not be loaded.")


def render_dataset_explorer() -> None:
    """Render Page 3: Interactive Dataset Explorer on bounded sample."""
    render_page_header(
        "Dataset Explorer",
        "Inspect network flow records, column dtypes, and filter traffic segments interactively",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        split = st.selectbox("Dataset Split", ["train", "test"], format_func=lambda x: "Training Set" if x == "train" else "Testing Set")
    with c2:
        sample_n = st.slider("Preview Sample Rows (for responsive UI)", min_value=500, max_value=10000, value=2500, step=500)

    df = load_sample_split(split=split, nrows=sample_n)
    if df is None:
        st.error(f"Could not load UNSW-NB15 {split}ing dataset CSV.")
        return

    st.caption(f"Displaying sample of first {len(df):,} rows from the predefined {split}ing set.")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        proto_filter = st.multiselect("Protocol (proto)", sorted(df["proto"].astype(str).unique().tolist()))
    with f2:
        service_filter = st.multiselect("Service", sorted(df["service"].astype(str).unique().tolist()))
    with f3:
        state_filter = st.multiselect("State", sorted(df["state"].astype(str).unique().tolist()))
    with f4:
        cat_filter = st.multiselect("Attack Category", sorted(df["attack_cat"].astype(str).unique().tolist()))

    filtered = df
    if proto_filter:
        filtered = filtered[filtered["proto"].astype(str).isin(proto_filter)]
    if service_filter:
        filtered = filtered[filtered["service"].astype(str).isin(service_filter)]
    if state_filter:
        filtered = filtered[filtered["state"].astype(str).isin(state_filter)]
    if cat_filter:
        filtered = filtered[filtered["attack_cat"].astype(str).isin(cat_filter)]

    st.write(f"**Filtered Rows:** {len(filtered):,} | **Columns:** {filtered.shape[1]}")
    st.dataframe(filtered.head(300), width="stretch", hide_index=True)

    with st.expander("Column Data Types & Sample Summary"):
        dtype_df = pd.DataFrame(
            {"Column": filtered.columns, "Dtype": [str(t) for t in filtered.dtypes]}
        )
        st.dataframe(dtype_df, width="stretch", hide_index=True)
