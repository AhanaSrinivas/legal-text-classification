"""Shared classification evaluation and artifact writing."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> dict:
    """Return aggregate metrics, per-class metrics with support, and confusion matrix."""
    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    aggregate = {}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        aggregate[f"{average}_precision"] = float(p)
        aggregate[f"{average}_recall"] = float(r)
        aggregate[f"{average}_f1"] = float(f)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **aggregate,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_confusion_matrix(metrics: dict, label_names: Sequence[str], path: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(label_names)),
        yticks=range(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col, row, int(matrix[row, col]),
                ha="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def metrics_row(model_name: str, split: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
