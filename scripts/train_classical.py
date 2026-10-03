"""Train and evaluate Phase 5 TF-IDF classical baselines."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import SCDB_LABEL_NAMES, load_scotus_dataset
from src.evaluate import evaluate_predictions, metrics_row, save_confusion_matrix
from src.preprocess import PreprocessingConfig, build_tfidf_vectorizer, fit_on_train


def fit_lr(x_train, y_train, c, class_weight):
    return LogisticRegression(
        C=c, class_weight=class_weight, max_iter=5, solver="saga",
        tol=0.5,
        random_state=42,
    ).fit(x_train, y_train)


def main():
    dataset = load_scotus_dataset()
    y_train = np.asarray(dataset["train"]["label"])
    y_val = np.asarray(dataset["validation"]["label"])
    y_test = np.asarray(dataset["test"]["label"])

    # Keep model matrices smaller than the full preprocessing benchmark on CPU.
    config = PreprocessingConfig(tfidf_ngram_range=(1, 2), tfidf_max_features=10_000)
    vectorizer = fit_on_train(build_tfidf_vectorizer(config), dataset["train"]["text"])
    x_train = vectorizer.transform(dataset["train"]["text"])
    x_val = vectorizer.transform(dataset["validation"]["text"])
    x_test = vectorizer.transform(dataset["test"]["text"])

    rows = []
    reports = {}
    selected = {}
    prediction_outputs = {}
    for name, candidates, fitter in [
        (
            "tfidf_logistic_regression",
            [(c, None) for c in (0.1, 1.0)],
            lambda c, weight: fit_lr(x_train, y_train, c, weight),
        ),
        (
            "tfidf_linear_svc",
            [(c,) for c in (0.01, 0.1, 1.0)],
            lambda c: LinearSVC(C=c, loss="squared_hinge", random_state=42).fit(
                x_train, y_train
            ),
        ),
    ]:
        best = None
        for candidate in candidates:
            model = fitter(*candidate)
            val_metrics = evaluate_predictions(
                y_val, model.predict(x_val), SCDB_LABEL_NAMES
            )
            if best is None or val_metrics["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"params": candidate, "metrics": val_metrics, "model": model}
        selected[name] = {"params": best["params"], "validation": best["metrics"]}

        test_pred = best["model"].predict(x_test)
        val_pred = best["model"].predict(x_val)
        if hasattr(best["model"], "predict_proba"):
            val_scores = best["model"].predict_proba(x_val)
            test_scores = best["model"].predict_proba(x_test)
            score_kind = "probabilities"
        else:
            val_scores = best["model"].decision_function(x_val)
            test_scores = best["model"].decision_function(x_test)
            score_kind = "decision_scores"
        prediction_outputs[name] = {
            "val_labels": y_val,
            "val_predictions": val_pred,
            "val_scores": val_scores,
            "test_labels": y_test,
            "test_predictions": test_pred,
            "test_scores": test_scores,
            "score_kind": score_kind,
        }
        test_metrics = evaluate_predictions(y_test, test_pred, SCDB_LABEL_NAMES)
        reports[name] = {"validation": best["metrics"], "test": test_metrics}
        rows.extend([
            metrics_row(name, "validation", best["metrics"]),
            metrics_row(name, "test", test_metrics),
        ])
        save_confusion_matrix(
            test_metrics, SCDB_LABEL_NAMES,
            f"results/figures/confusion_matrix_{name}.png",
        )

    # Validation-only numeric-token ablation; it never evaluates on test.
    ablation_rows = []
    for drop_numeric in (True, False):
        ablation_config = PreprocessingConfig(
            tfidf_ngram_range=(1, 2), tfidf_max_features=10_000,
            drop_numeric_tokens=drop_numeric
        )
        ablation_vectorizer = fit_on_train(
            build_tfidf_vectorizer(ablation_config), dataset["train"]["text"]
        )
        ablation_model = fit_lr(
            ablation_vectorizer.transform(dataset["train"]["text"]),
            y_train, 1.0, None,
        )
        ablation_metrics = evaluate_predictions(
            y_val,
            ablation_model.predict(
                ablation_vectorizer.transform(dataset["validation"]["text"])
            ),
            SCDB_LABEL_NAMES,
        )
        ablation_rows.append({
            "drop_numeric_tokens": drop_numeric,
            **metrics_row("tfidf_lr_numeric_ablation", "validation", ablation_metrics),
        })

    majority = json.loads(Path("results/majority_baseline.json").read_text())
    majority_metrics = {
        "accuracy": majority["test_accuracy"],
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": majority["test_macro_f1"],
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": majority["test_weighted_f1"],
    }
    rows.append(metrics_row("majority_baseline", "test", majority_metrics))

    transformer_info = json.loads(Path("results/transformer_run_info.json").read_text())
    if transformer_info.get("smoke_test") is False:
        predictions = np.load("results/transformer_predictions.npz")
        transformer_metrics = evaluate_predictions(
            predictions["test_labels"], predictions["test_predictions"], SCDB_LABEL_NAMES
        )
        reports["transformer_legal_bert"] = {"test": transformer_metrics}
        rows.append(metrics_row("transformer_legal_bert", "test", transformer_metrics))
        save_confusion_matrix(
            transformer_metrics, SCDB_LABEL_NAMES,
            "results/figures/confusion_matrix_transformer_legal_bert.png",
        )

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)
    Path("results/classification_reports").mkdir(exist_ok=True)
    Path("results/classification_reports/classical_reports.json").write_text(
        json.dumps(reports, indent=2)
    )
    Path("results/classical_selection.json").write_text(
        json.dumps(selected, indent=2)
    )
    pd.DataFrame(ablation_rows).to_csv("results/numeric_token_ablation_validation.csv", index=False)
    Path("results/predictions").mkdir(exist_ok=True)
    for name, output in prediction_outputs.items():
        score_kind = output.pop("score_kind")
        np.savez_compressed(
            f"results/predictions/{name}.npz",
            **output,
            score_kind=np.array(score_kind),
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
