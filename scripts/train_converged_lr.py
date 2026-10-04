"""Train the bounded, better-converged TF-IDF logistic-style baseline.

This script reuses the Phase 5 train-fitted TF-IDF configuration and tunes
SGDClassifier alpha on validation macro F1 only. The test split is predicted
once for the selected model.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import SCDB_LABEL_NAMES, load_scotus_dataset
from src.evaluate import evaluate_predictions, metrics_row
from src.preprocess import PreprocessingConfig, build_tfidf_vectorizer, fit_on_train


def main():
    started = time.perf_counter()
    dataset = load_scotus_dataset()
    y_train = np.asarray(dataset["train"]["label"])
    y_val = np.asarray(dataset["validation"]["label"])
    y_test = np.asarray(dataset["test"]["label"])

    config = PreprocessingConfig(tfidf_ngram_range=(1, 2), tfidf_max_features=10_000)
    vectorizer = fit_on_train(build_tfidf_vectorizer(config), dataset["train"]["text"])
    x_train = vectorizer.transform(dataset["train"]["text"])
    x_val = vectorizer.transform(dataset["validation"]["text"])
    x_test = vectorizer.transform(dataset["test"]["text"])

    candidates = {}
    for alpha in (1e-6, 1e-5, 1e-4):
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=50,
            tol=1e-3,
            random_state=42,
            class_weight=None,
        )
        model.fit(x_train, y_train)
        validation = evaluate_predictions(
            y_val, model.predict(x_val), SCDB_LABEL_NAMES
        )
        candidates[str(alpha)] = {
            "validation": validation,
            "n_iter": int(model.n_iter_),
            "converged": bool(model.n_iter_ < model.max_iter),
        }
        candidates[str(alpha)]["_model"] = model

    selected_alpha = max(
        candidates,
        key=lambda alpha: candidates[alpha]["validation"]["macro_f1"],
    )
    selected = candidates[selected_alpha]
    model = selected["_model"]
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_scores = model.decision_function(x_val)
    test_scores = model.decision_function(x_test)
    val_metrics = evaluate_predictions(y_val, val_pred, SCDB_LABEL_NAMES)
    test_metrics = evaluate_predictions(y_test, test_pred, SCDB_LABEL_NAMES)

    output = {
        "val_labels": y_val,
        "val_predictions": val_pred,
        "val_scores": val_scores,
        "test_labels": y_test,
        "test_predictions": test_pred,
        "test_scores": test_scores,
        "score_kind": np.array("decision_scores"),
    }
    np.savez_compressed("results/predictions/tfidf_logistic_regression.npz", **output)

    metadata = {
        "model": "tfidf_logistic_regression",
        "estimator": "SGDClassifier",
        "loss": "log_loss",
        "penalty": "l2",
        "max_iter": 50,
        "tol": 1e-3,
        "class_weight": None,
        "random_state": 42,
        "alpha_grid": [1e-6, 1e-5, 1e-4],
        "selected_alpha": float(selected_alpha),
        "candidates": {
            alpha: {key: value for key, value in details.items() if key != "_model"}
            for alpha, details in candidates.items()
        },
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path("results/converged_lr_selection.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps({
        "selected_alpha": float(selected_alpha),
        "n_iter": int(model.n_iter_),
        "converged": bool(model.n_iter_ < model.max_iter),
        "elapsed_seconds": metadata["elapsed_seconds"],
        "validation_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
