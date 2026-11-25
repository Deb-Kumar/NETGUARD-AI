"""NETGUARD AI Streamlit Web Dashboard Entrypoint (16 Pages)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root resolves `app` as the package rather than `app/app.py`
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path = [p for p in sys.path if Path(p).resolve() != APP_DIR]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pages.batch_prediction import render_batch_prediction  # noqa: E402
from app.pages.cv_tuning_comparison import (  # noqa: E402
    render_cross_validation,
    render_hyperparameter_tuning,
    render_model_comparison,
)
from app.pages.dashboard import render_dashboard  # noqa: E402
from app.pages.dataset_views import (  # noqa: E402
    render_dataset_explorer,
    render_dataset_management,
)
from app.pages.eda_preprocessing import (  # noqa: E402
    render_data_preprocessing,
    render_eda_analytics,
)
from app.pages.history_reports_system import (  # noqa: E402
    render_about_project,
    render_prediction_history,
    render_reports_exports,
    render_settings,
)
from app.pages.single_prediction import render_single_prediction  # noqa: E402
from app.pages.training_eval_views import (  # noqa: E402
    render_model_evaluation,
    render_model_training,
)
from app.styles.theme import apply_cyber_theme  # noqa: E402
from src.utils.config import APP_VERSION, PROJECT_NAME  # noqa: E402


def main() -> None:
    """Configure Streamlit page layout, theme, and multi-page navigation."""
    st.set_page_config(
        page_title=f"{PROJECT_NAME} | Network Intrusion Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_cyber_theme()

    st.sidebar.markdown(
        f"## 🛡️ {PROJECT_NAME}\n"
        f"**Version:** `v{APP_VERSION}`  \n"
        f"Network Intrusion Detection System"
    )

    pages = {
        "Command Center": [
            st.Page(render_dashboard, title="Dashboard", icon="📊", url_path="dashboard", default=True),
        ],
        "Dataset & Analytics": [
            st.Page(render_dataset_management, title="Dataset Management", icon="🗂️", url_path="dataset-management"),
            st.Page(render_dataset_explorer, title="Dataset Explorer", icon="🔍", url_path="dataset-explorer"),
            st.Page(render_eda_analytics, title="EDA and Analytics", icon="📈", url_path="eda-analytics"),
            st.Page(render_data_preprocessing, title="Data Preprocessing", icon="⚙️", url_path="data-preprocessing"),
        ],
        "ML Pipeline": [
            st.Page(render_model_training, title="Model Training", icon="🧠", url_path="model-training"),
            st.Page(render_model_evaluation, title="Model Evaluation", icon="🎯", url_path="model-evaluation"),
            st.Page(render_cross_validation, title="Cross-Validation", icon="🔁", url_path="cross-validation"),
            st.Page(render_hyperparameter_tuning, title="Hyperparameter Tuning", icon="🎛️", url_path="hyperparameter-tuning"),
            st.Page(render_model_comparison, title="Model Comparison", icon="🏆", url_path="model-comparison"),
        ],
        "Threat Prediction": [
            st.Page(render_single_prediction, title="Single Traffic Prediction", icon="🔎", url_path="single-prediction"),
            st.Page(render_batch_prediction, title="Batch CSV Prediction", icon="📂", url_path="batch-prediction"),
        ],
        "Logs, Reports & System": [
            st.Page(render_prediction_history, title="Prediction History", icon="📜", url_path="prediction-history"),
            st.Page(render_reports_exports, title="Reports and Exports", icon="📑", url_path="reports-exports"),
            st.Page(render_about_project, title="About Project", icon="ℹ️", url_path="about-project"),
            st.Page(render_settings, title="Settings", icon="🛠️", url_path="settings"),
        ],
    }

    selected_page = st.navigation(pages)
    selected_page.run()


if __name__ == "__main__":
    main()
