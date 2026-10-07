import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from src import demo
from src.data import SCDB_LABEL_NAMES
from src.preprocess import PreprocessingConfig, build_tfidf_vectorizer, fit_on_train


def _toy_vectorizer_and_data():
    texts = [f"issue area {name.lower()} words" for name in SCDB_LABEL_NAMES] * 2
    labels = np.array(list(range(13)) * 2)
    vectorizer = fit_on_train(build_tfidf_vectorizer(PreprocessingConfig(min_df=1)), texts)
    return vectorizer, vectorizer.transform(texts), labels


def test_top_k_orders_scores_descending():
    scores = np.zeros(13)
    scores[[3, 7, 11]] = [0.2, 0.5, 0.3]
    table = demo.top_k(scores, k=3)
    assert list(table["Issue area"]) == ["Economic Activity", "Federal Taxation", "Due Process"]
    assert list(table["Score"]) == [0.5, 0.3, 0.2]


def test_truncation_note_only_for_long_inputs():
    assert demo.truncation_note(100) is None
    assert demo.truncation_note(510) is None
    note = demo.truncation_note(5100)
    assert "first 512 tokens" in note and "10%" in note


def test_linear_svc_returns_decision_scores():
    vectorizer, x, y = _toy_vectorizer_and_data()
    svc = LinearSVC(random_state=42).fit(x, y)
    result = demo.predict_sklearn(vectorizer, svc, "issue area federal taxation words")
    assert result["score_kind"] == "decision score"
    assert result["scores"].shape == (13,)
    assert SCDB_LABEL_NAMES[result["label"]] == "Federal Taxation"


def test_log_loss_model_returns_probabilities():
    vectorizer, x, y = _toy_vectorizer_and_data()
    sgd = SGDClassifier(loss="log_loss", random_state=42).fit(x, y)
    result = demo.predict_sklearn(vectorizer, sgd, "issue area unions words")
    assert result["score_kind"] == "probability"
    assert result["scores"].sum() == pytest.approx(1.0)


def test_reported_setting_note_flags_only_mismatches():
    selected_c = json.loads((demo.RESULTS_DIR / "classical_selection.json").read_text())[
        "tfidf_linear_svc"]["params"][0]
    assert demo.reported_setting_note("tfidf_linear_svc", LinearSVC(C=selected_c)) is None
    note = demo.reported_setting_note("tfidf_linear_svc", LinearSVC(C=selected_c / 10))
    assert note and f"C={selected_c}" in note
    alpha = json.loads((demo.RESULTS_DIR / "converged_lr_selection.json").read_text())["selected_alpha"]
    assert demo.reported_setting_note("tfidf_logistic_regression", SGDClassifier(alpha=alpha)) is None


def test_no_artifacts_means_no_live_models(monkeypatch, tmp_path):
    monkeypatch.setenv("LTC_MODELS_DIR", str(tmp_path))
    assert demo.available_models() == []
    assert demo.load_sklearn_model("tfidf_linear_svc") is None
    assert demo.load_bert() is None


def test_metrics_table_reads_results_and_sorts_by_macro_f1():
    table = demo.metrics_table("test")
    saved = pd.read_csv(demo.RESULTS_DIR / "metrics.csv")
    saved = saved[saved["split"] == "test"]
    assert len(table) == len(saved)
    assert table["Macro F1"].tolist() == sorted(saved["macro_f1"], reverse=True)
    assert table["Macro F1 95% CI"].str.startswith("[").all()
    assert "Macro F1 95% CI" not in demo.metrics_table("validation")


def test_stored_predictions_match_archives():
    case = demo.load_sample_cases()[0]
    table = demo.stored_predictions(case["test_index"])
    assert len(table) == 5
    archive = np.load(demo.RESULTS_DIR / "predictions" / "tfidf_linear_svc.npz")
    predicted = SCDB_LABEL_NAMES[int(archive["test_predictions"][case["test_index"]])]
    assert table.iloc[0]["Stored prediction"] == predicted
    assert int(archive["test_labels"][case["test_index"]]) == case["label_id"]


def test_per_class_table_includes_support():
    table = demo.per_class_table("transformer_legal_bert")
    assert len(table) == 13
    assert table["Test support"].sum() == 1400


needs_tfidf = pytest.mark.skipif(
    "tfidf_linear_svc" not in demo.available_models(),
    reason="TF-IDF demo artifacts not downloaded",
)


@needs_tfidf
@pytest.mark.parametrize("name", ["tfidf_linear_svc", "tfidf_logistic_regression"])
def test_downloaded_tfidf_model_predicts_sample(name):
    vectorizer, classifier = demo.load_sklearn_model(name)
    assert list(classifier.classes_) == list(range(13))
    case = demo.load_sample_cases()[0]
    result = demo.predict_sklearn(vectorizer, classifier, case["full_text"])
    assert 0 <= result["label"] < 13
    assert result["vocabulary_hits"] > 0


@pytest.mark.skipif(
    os.getenv("LTC_RUN_BERT") != "1" or "transformer_legal_bert" not in demo.available_models(),
    reason="set LTC_RUN_BERT=1 with models/legal_bert_state.pt present (slow, needs the tokenizer)",
)
def test_downloaded_legal_bert_predicts_sample():
    tokenizer, model = demo.load_bert()
    case = demo.load_sample_cases()[0]
    result = demo.predict_bert(tokenizer, model, case["full_text"])
    assert result["scores"].sum() == pytest.approx(1.0, abs=1e-5)
    assert demo.count_bert_tokens(tokenizer, case["full_text"]) > 0
