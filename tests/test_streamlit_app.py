"""Phase 12 tests: verify Streamlit multipage app and all 16 page views render cleanly."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.utils.config import get_paths


def test_streamlit_app_entrypoint_loads_without_exception() -> None:
    app_path = get_paths().app_dir / "app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception


@pytest.mark.parametrize(
    ("module_name", "func_name"),
    [
        ("app.pages.dashboard", "render_dashboard"),
        ("app.pages.dataset_views", "render_dataset_management"),
        ("app.pages.dataset_views", "render_dataset_explorer"),
        ("app.pages.eda_preprocessing", "render_eda_analytics"),
        ("app.pages.eda_preprocessing", "render_data_preprocessing"),
        ("app.pages.training_eval_views", "render_model_training"),
        ("app.pages.training_eval_views", "render_model_evaluation"),
        ("app.pages.cv_tuning_comparison", "render_cross_validation"),
        ("app.pages.cv_tuning_comparison", "render_hyperparameter_tuning"),
        ("app.pages.cv_tuning_comparison", "render_model_comparison"),
        ("app.pages.single_prediction", "render_single_prediction"),
        ("app.pages.batch_prediction", "render_batch_prediction"),
        ("app.pages.history_reports_system", "render_prediction_history"),
        ("app.pages.history_reports_system", "render_reports_exports"),
        ("app.pages.history_reports_system", "render_about_project"),
        ("app.pages.history_reports_system", "render_settings"),
    ],
)
def test_all_16_views_render_without_exception(module_name: str, func_name: str) -> None:
    root_str = repr(str(get_paths().root))
    script = f"""
import importlib
import sys
root = {root_str}
if root not in sys.path:
    sys.path.insert(0, root)
from app.styles.theme import apply_cyber_theme
apply_cyber_theme()
getattr(importlib.import_module({module_name!r}), {func_name!r})()
"""
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert not at.exception


def test_stratified_ui_samples_avoid_sequential_class_skew() -> None:
    from app.components.data_access import load_stratified_split_sample, load_stratified_test_sample

    train_sample = load_stratified_split_sample(split="train", nrows=2500)
    assert train_sample is not None
    assert set(train_sample["label"].unique()) == {0, 1}
    assert train_sample["attack_cat"].nunique() == 10
    assert train_sample["label"].head(300).nunique() == 2

    test_sample = load_stratified_test_sample(nrows=400)
    assert test_sample is not None
    assert set(test_sample["label"].unique()) == {0, 1}
    assert test_sample["attack_cat"].nunique() == 10
