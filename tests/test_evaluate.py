import numpy as np
import pytest

from src.evaluate import evaluate_predictions, metrics_row


LABELS = ["a", "b", "c"]
# Hand-worked example:
#   class a: TP=1 FP=1 FN=1 -> P=1/2, R=1/2, F1=1/2
#   class b: TP=2 FP=1 FN=0 -> P=2/3, R=1,   F1=4/5
#   class c: TP=0 FP=0 FN=1 -> P=0,   R=0,   F1=0 (zero_division=0)
Y_TRUE = [0, 0, 1, 1, 2]
Y_PRED = [0, 1, 1, 1, 0]


def test_metrics_match_hand_calculation():
    metrics = evaluate_predictions(Y_TRUE, Y_PRED, LABELS)
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["macro_f1"] == pytest.approx((1 / 2 + 4 / 5 + 0) / 3)
    assert metrics["weighted_f1"] == pytest.approx((2 * 1 / 2 + 2 * 4 / 5 + 1 * 0) / 5)
    assert metrics["macro_precision"] == pytest.approx((1 / 2 + 2 / 3 + 0) / 3)
    assert metrics["macro_recall"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert metrics["per_class"]["b"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 4 / 5, "support": 2}
    )
    assert metrics["per_class"]["c"]["f1"] == 0.0


def test_confusion_matrix_rows_are_true_labels():
    matrix = evaluate_predictions(Y_TRUE, Y_PRED, LABELS)["confusion_matrix"]
    assert matrix == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_unpredicted_and_absent_classes_still_reported():
    metrics = evaluate_predictions([0, 0], [0, 0], LABELS)
    assert set(metrics["per_class"]) == set(LABELS)
    assert metrics["per_class"]["c"]["support"] == 0
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)


def test_macro_f1_penalises_ignoring_a_rare_class():
    y_true = [0] * 9 + [1]
    always_majority = evaluate_predictions(y_true, [0] * 10, ["big", "rare"])
    assert always_majority["accuracy"] == pytest.approx(0.9)
    assert always_majority["macro_f1"] < 0.5


def test_metrics_row_has_csv_columns():
    row = metrics_row("m", "test", evaluate_predictions(Y_TRUE, Y_PRED, LABELS))
    assert list(row) == [
        "model", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
