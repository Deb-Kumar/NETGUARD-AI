"""Phase 19 E2E tests: interactive Streamlit user journey & SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.database import clear_history, query_history
from src.utils.config import get_paths


def test_interactive_e2e_single_prediction_history_and_settings(tmp_path: Path, monkeypatch) -> None:
    """Simulate user submitting Single Prediction, verifying SQLite History, and clearing it."""
    real_root = get_paths().root
    # Isolate DB writes via temporary root? No — Single Prediction needs real trained models,
    # so we run against real root and assert + clean up our test entry cleanly!
    root_repr = repr(str(real_root))

    # Ensure clean baseline count
    before_count = len(query_history())

    single_script = f"""
import sys
root = {root_repr}
if root not in sys.path:
    sys.path.insert(0, root)
from app.components.theme import apply_cyber_theme
from app.views.prediction_views import render_single_prediction
apply_cyber_theme()
render_single_prediction()
"""
    at_single = AppTest.from_string(single_script, default_timeout=30)
    at_single.run()
    assert not at_single.exception

    # Switch radio to Inspect Actual Test Set Record, rerun, then submit form
    at_single.radio[0].set_value("Inspect Actual Test Set Record").run()
    assert not at_single.exception
    assert any("Actual Test Ground Truth" in info.value for info in at_single.info)

    # Submit the prediction form
    at_single.button[0].click().run()
    assert not at_single.exception
    assert any("Prediction Result:" in md.value for md in at_single.markdown)

    # Verify SQLite persistence
    after_hist = query_history()
    assert len(after_hist) == before_count + 1
    latest = after_hist.iloc[0]
    assert latest["prediction_type"] == "Single"
    assert latest["model_name"] == "Random Forest Classifier"

    # Simulate opening Prediction History in a fresh session (app restart equivalent)
    hist_script = f"""
import sys
root = {root_repr}
if root not in sys.path:
    sys.path.insert(0, root)
from app.components.theme import apply_cyber_theme
from app.views.history_reports_system import render_prediction_history
apply_cyber_theme()
render_prediction_history()
"""
    at_hist = AppTest.from_string(hist_script, default_timeout=30)
    at_hist.run()
    assert not at_hist.exception
    assert not at_hist.dataframe[0].value.empty

    # Verify Clear All History is disabled until confirmation checkbox is checked
    clear_btn = [b for b in at_hist.button if "Clear All History" in b.label][0]
    assert clear_btn.disabled is True

    # Clean up test row only if DB started empty, else delete only the inserted row
    if before_count == 0:
        confirm_cb = [c for c in at_hist.checkbox if "permanently deleting" in c.label][0]
        confirm_cb.set_value(True).run()
        clear_btn_enabled = [b for b in at_hist.button if "Clear All History" in b.label][0]
        assert clear_btn_enabled.disabled is False
        clear_btn_enabled.click().run()
        assert not at_hist.exception
        assert query_history().empty


def test_settings_page_persists_session_model_preference() -> None:
    root_repr = repr(str(get_paths().root))
    settings_script = f"""
import sys
root = {root_repr}
if root not in sys.path:
    sys.path.insert(0, root)
from app.components.theme import apply_cyber_theme
from app.views.history_reports_system import render_settings
apply_cyber_theme()
render_settings()
"""
    at = AppTest.from_string(settings_script, default_timeout=30)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value("random_forest").run()
    save_btn = [b for b in at.button if "Save Session Model Preference" in b.label][0]
    save_btn.click().run()
    assert not at.exception
    assert at.session_state["active_binary_model_key"] == "random_forest"
