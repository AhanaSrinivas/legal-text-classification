"""Rebuild Phase 8 evaluation artifacts from saved predictions only."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import SCDB_LABEL_NAMES, load_scotus_dataset
from src.evaluate import evaluate_predictions, metrics_row


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = ROOT / "results" / "predictions"


def load_archive(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def add_archive(name, archive):
    return {
        "name": name,
        "validation_labels": np.asarray(archive["val_labels"]),
        "validation_predictions": np.asarray(archive["val_predictions"]),
        "validation_scores": np.asarray(
            archive.get("val_scores", archive.get("val_probabilities"))
        ),
        "test_labels": np.asarray(archive["test_labels"]),
        "test_predictions": np.asarray(archive["test_predictions"]),
        "test_scores": np.asarray(
            archive.get("test_scores", archive.get("test_probabilities"))
        ),
    }


def save_normalized_confusion(metrics, name):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set(
        title=f"Normalized confusion matrix: {name}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(SCDB_LABEL_NAMES)),
        yticks=range(len(SCDB_LABEL_NAMES)),
        xticklabels=SCDB_LABEL_NAMES,
        yticklabels=SCDB_LABEL_NAMES,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    path = ROOT / "results" / "figures" / f"confusion_matrix_normalized_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bootstrap_macro_f1(y_true, y_pred, seed=42, resamples=1000):
    rng = np.random.default_rng(seed)
    values = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(resamples):
        indices = rng.integers(0, len(y_true), len(y_true))
        values.append(
            evaluate_predictions(y_true[indices], y_pred[indices], SCDB_LABEL_NAMES)["macro_f1"]
        )
    return {
        "resamples": resamples,
        "seed": seed,
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def confusion_pairs(metrics):
    matrix = np.asarray(metrics["confusion_matrix"])
    pairs = []
    for true_idx in range(matrix.shape[0]):
        for pred_idx in range(matrix.shape[1]):
            if true_idx != pred_idx and matrix[true_idx, pred_idx]:
                pairs.append((int(matrix[true_idx, pred_idx]), SCDB_LABEL_NAMES[true_idx],
                              SCDB_LABEL_NAMES[pred_idx]))
    return sorted(pairs, reverse=True)


def write_error_analysis(models, dataset, test_metrics, val_metrics):
    svc = models["tfidf_linear_svc"]
    bert = models["transformer_legal_bert"]
    y_test = svc["test_labels"]
    svc_pred = svc["test_predictions"]
    bert_pred = bert["test_predictions"]
    texts = dataset["test"]["text"]
    word_counts = np.asarray([len(str(text).split()) for text in texts])
    quartiles = np.percentile(word_counts, [25, 50, 75])

    def quartile_rows(pred):
        rows = []
        bounds = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
                  (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]
        for index, (low, high) in enumerate(bounds, 1):
            mask = (word_counts >= low) & (word_counts <= high if index == 4 else word_counts < high)
            rows.append((index, int(mask.sum()), float(np.mean(pred[mask] == y_test[mask]))))
        return rows

    lines = [
        "# Error analysis",
        "",
        "This analysis uses the saved validation/test predictions and the test text. Explanations are hypotheses, not established causes.",
        "",
        "## Most confused test-class pairs",
    ]
    for name in ("tfidf_linear_svc", "transformer_legal_bert"):
        lines.append(f"\n### {name}")
        lines.extend(f"- {count}: {true} -> {pred}" for count, true, pred in confusion_pairs(test_metrics[name])[:5])
        lines.append("Lowest-F1 classes: " + "; ".join(
            f"{label} (F1={values['f1']:.4f}, support={values['support']})"
            for label, values in sorted(test_metrics[name]["per_class"].items(), key=lambda item: item[1]["f1"])[:5]
        ))

    lines += ["", "## Accuracy by test document-length quartile"]
    lines.append("| Model | Quartile | Documents | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for name, pred in (("tfidf_linear_svc", svc_pred), ("transformer_legal_bert", bert_pred)):
        for quartile, count, accuracy in quartile_rows(pred):
            lines.append(f"| {name} | {quartile} | {count} | {accuracy:.4f} |")

    only_svc = (svc_pred == y_test) & (bert_pred != y_test)
    only_bert = (svc_pred != y_test) & (bert_pred == y_test)
    both_wrong = (svc_pred != y_test) & (bert_pred != y_test)
    lines += [
        "",
        "## Correctness overlap",
        f"- Only SVC correct: {int(only_svc.sum())}",
        f"- Only Legal-BERT correct: {int(only_bert.sum())}",
        f"- Both wrong: {int(both_wrong.sum())}",
        "",
        "## Concrete misclassified examples",
    ]
    candidate_indices = np.flatnonzero((svc_pred != y_test) | (bert_pred != y_test))[:10]
    for number, index in enumerate(candidate_indices, 1):
        lines += [
            f"\n### Example {number} (test index {int(index)})",
            f"- True: {SCDB_LABEL_NAMES[int(y_test[index])]}; SVC: {SCDB_LABEL_NAMES[int(svc_pred[index])]}; Legal-BERT: {SCDB_LABEL_NAMES[int(bert_pred[index])]}.",
            f"- Text (first 300 characters): {str(texts[int(index)])[:300].replace(chr(10), ' ')}",
            "- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.",
        ]

    lines += ["", "## Validation-to-test macro-F1 change"]
    for name in ("tfidf_linear_svc", "transformer_legal_bert"):
        drop = val_metrics[name]["macro_f1"] - test_metrics[name]["macro_f1"]
        lines.append(f"- {name}: validation {val_metrics[name]['macro_f1']:.6f} to test {test_metrics[name]['macro_f1']:.6f} (change {drop:.6f}).")
    lines += [
        "",
        "## Qualitative comparison with the paper",
        "The paper's reported ordering is used only qualitatively. Its dataset construction, split, labels, preprocessing, and reported settings differ from this project; its transformer experiment reported no metric. Therefore these results are not a numerical reproduction of the paper.",
    ]
    (ROOT / "results" / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    dataset = load_scotus_dataset()
    models = {}
    for path in sorted(PREDICTION_DIR.glob("*.npz")):
        models[path.stem] = add_archive(path.stem, load_archive(path))

    info = json.loads((ROOT / "results" / "transformer_run_info.json").read_text())
    if info.get("smoke_test") is not False:
        raise RuntimeError("Refusing to evaluate transformer predictions from a smoke test.")
    transformer = add_archive("transformer_legal_bert", load_archive(ROOT / "results" / "transformer_predictions.npz"))
    models["transformer_legal_bert"] = transformer
    np.savez_compressed(
        PREDICTION_DIR / "transformer_legal_bert.npz",
        val_labels=transformer["validation_labels"],
        val_predictions=transformer["validation_predictions"],
        val_scores=transformer["validation_scores"],
        test_labels=transformer["test_labels"],
        test_predictions=transformer["test_predictions"],
        test_scores=transformer["test_scores"],
        score_kind=np.array("probabilities"),
    )

    majority = int(np.bincount(np.asarray(dataset["train"]["label"])).argmax())
    for split in ("validation", "test"):
        labels = np.asarray(dataset[split]["label"])
        predictions = np.full(labels.shape, majority)
        key = "validation" if split == "validation" else split
        models.setdefault("majority_baseline", {})[f"{key}_labels"] = labels
        models["majority_baseline"][f"{key}_predictions"] = predictions
        models["majority_baseline"][f"{key}_scores"] = np.zeros((len(labels), len(SCDB_LABEL_NAMES)))

    rows = []
    all_metrics = {}
    per_class_rows = []
    ci = {}
    for name, model in models.items():
        all_metrics[name] = {}
        for split in ("validation", "test"):
            metrics = evaluate_predictions(model[f"{split}_labels"], model[f"{split}_predictions"], SCDB_LABEL_NAMES)
            all_metrics[name][split] = metrics
            rows.append(metrics_row(name, split, metrics))
            if split == "test":
                for label, values in metrics["per_class"].items():
                    per_class_rows.append({"class": label, "model": name, **values})
        save_normalized_confusion(all_metrics[name]["test"], name)
        ci[name] = bootstrap_macro_f1(model["test_labels"], model["test_predictions"]) 

    pd.DataFrame(rows).to_csv(ROOT / "results" / "metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(ROOT / "results" / "per_class_f1.csv", index=False)
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")

    comparison = pd.DataFrame([
        {"model": name, "test_macro_f1": values["test"]["macro_f1"], "test_accuracy": values["test"]["accuracy"]}
        for name, values in all_metrics.items()
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.set_index("model")[["test_macro_f1", "test_accuracy"]].plot.bar(ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Test model comparison (per-class support is in per_class_f1.csv)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "model_comparison.png", dpi=180)
    plt.close(fig)

    write_error_analysis(models, dataset, {name: values["test"] for name, values in all_metrics.items()},
                         {name: values["validation"] for name, values in all_metrics.items()})
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
