"""Saved results must be reproducible from the saved prediction archives.

These tests need no dataset download: every archive stores its own labels.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import SCDB_LABEL_NAMES
from src.evaluate import evaluate_predictions


RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = pd.read_csv(RESULTS / "metrics.csv")
ARCHIVES = sorted((RESULTS / "predictions").glob("*.npz"))
COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_precision", "weighted_recall", "weighted_f1"]


def _row(model, split):
    rows = METRICS[(METRICS["model"] == model) & (METRICS["split"] == split)]
    assert len(rows) == 1, f"expected one row for {model}/{split}"
    return rows.iloc[0]


def test_every_model_has_validation_and_test_rows():
    for model in [p.stem for p in ARCHIVES] + ["majority_baseline"]:
        assert set(METRICS[METRICS["model"] == model]["split"]) == {"validation", "test"}


@pytest.mark.parametrize("archive_path", ARCHIVES, ids=lambda p: p.stem)
@pytest.mark.parametrize("split,prefix", [("validation", "val"), ("test", "test")])
def test_metrics_csv_recomputes_from_predictions(archive_path, split, prefix):
    archive = np.load(archive_path, allow_pickle=True)
    metrics = evaluate_predictions(
        archive[f"{prefix}_labels"], archive[f"{prefix}_predictions"], SCDB_LABEL_NAMES
    )
    row = _row(archive_path.stem, split)
    for column in COLUMNS:
        assert metrics[column] == pytest.approx(row[column], abs=1e-12), column


def test_majority_baseline_recomputes():
    labels = np.load(RESULTS / "predictions" / "tfidf_linear_svc.npz")["test_labels"]
    baseline = json.loads((RESULTS / "majority_baseline.json").read_text())
    predictions = np.full_like(labels, baseline["train_majority_class_id"])
    metrics = evaluate_predictions(labels, predictions, SCDB_LABEL_NAMES)
    assert metrics["macro_f1"] == pytest.approx(_row("majority_baseline", "test")["macro_f1"])
    assert metrics["accuracy"] == pytest.approx(baseline["test_accuracy"])


def test_per_class_csv_matches_test_predictions():
    per_class = pd.read_csv(RESULTS / "per_class_f1.csv")
    for archive_path in ARCHIVES:
        archive = np.load(archive_path, allow_pickle=True)
        metrics = evaluate_predictions(
            archive["test_labels"], archive["test_predictions"], SCDB_LABEL_NAMES
        )
        saved = per_class[per_class["model"] == archive_path.stem].set_index("class")
        for name, values in metrics["per_class"].items():
            assert saved.loc[name, "f1"] == pytest.approx(values["f1"], abs=1e-12)
            assert saved.loc[name, "support"] == values["support"]


def test_reported_transformer_archive_is_the_colab_run():
    original = np.load(RESULTS / "transformer_predictions.npz")
    copied = np.load(RESULTS / "predictions" / "transformer_legal_bert.npz", allow_pickle=True)
    np.testing.assert_array_equal(original["test_predictions"], copied["test_predictions"])
    info = json.loads((RESULTS / "transformer_run_info.json").read_text())
    assert info["smoke_test"] is False
    assert info["test_macro_f1"] == pytest.approx(_row("transformer_legal_bert", "test")["macro_f1"])


def test_bootstrap_intervals_contain_point_estimates():
    ci = json.loads((RESULTS / "bootstrap_ci.json").read_text())
    for model, bounds in ci.items():
        point = _row(model, "test")["macro_f1"]
        assert bounds["lower_95"] <= point <= bounds["upper_95"], model


def test_labels_are_identical_across_archives():
    reference = np.load(ARCHIVES[0], allow_pickle=True)
    for archive_path in ARCHIVES[1:]:
        archive = np.load(archive_path, allow_pickle=True)
        np.testing.assert_array_equal(reference["test_labels"], archive["test_labels"])
        np.testing.assert_array_equal(reference["val_labels"], archive["val_labels"])
