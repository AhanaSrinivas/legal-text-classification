"""Headless Streamlit smoke tests for app.py."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import demo


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    return app


def test_app_starts_without_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("LTC_MODELS_DIR", str(tmp_path))
    app = _run()
    sidebar_text = " ".join(md.value for md in app.sidebar.markdown)
    assert "missing" in sidebar_text
    app.button[0].click().run()
    assert not app.exception
    assert any("Select at least one model" in w.value for w in app.warning)


def test_results_tab_switches_split(monkeypatch, tmp_path):
    monkeypatch.setenv("LTC_MODELS_DIR", str(tmp_path))
    app = _run()
    split = next(r for r in app.radio if r.label == "Split")
    split.set_value("validation").run()
    assert not app.exception


def test_empty_custom_text_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("LTC_MODELS_DIR", str(tmp_path))
    app = _run()
    next(r for r in app.radio if r.label == "Input").set_value("Paste your own text").run()
    app.button[0].click().run()
    assert any("Enter some text" in w.value for w in app.warning)


@pytest.mark.skipif(
    "tfidf_linear_svc" not in demo.available_models(),
    reason="TF-IDF demo artifacts not downloaded",
)
def test_app_classifies_sample_with_tfidf_models():
    app = _run()
    app.button[0].click().run()
    assert not app.exception
    predicted = [m.value for m in app.metric]
    assert len(predicted) == 2
    assert all(value in demo.SCDB_LABEL_NAMES for value in predicted)
