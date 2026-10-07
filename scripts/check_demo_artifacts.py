"""Compare the downloaded demo artifacts with the reported test predictions.

Read-only diagnostic: nothing is trained. It answers "is the live demo model
the reported model?" and writes results/demo_artifact_check.json.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import SCDB_LABEL_NAMES, load_scotus_dataset
from src.demo import artifact_paths
from src.evaluate import evaluate_predictions


ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = artifact_paths()
    missing = [str(p) for k, p in paths.items() if k != "transformer_legal_bert" and not p.exists()]
    if missing:
        sys.exit(f"Demo artifacts missing: {missing}. Run scripts/download_artifacts.py first.")

    dataset = load_scotus_dataset()
    labels = np.asarray(dataset["test"]["label"])
    vectorizer = joblib.load(paths["tfidf_vectorizer"])
    features = vectorizer.transform(dataset["test"]["text"])

    report = {"sklearn_version_used": sklearn.__version__}
    for name in ("tfidf_linear_svc", "tfidf_logistic_regression"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            classifier = joblib.load(paths[name])
        pickled = {str(w.message.original_sklearn_version) for w in caught
                   if issubclass(w.category, InconsistentVersionWarning)}
        predictions = classifier.predict(features)
        stored = np.load(ROOT / "results" / "predictions" / f"{name}.npz")["test_predictions"]
        metrics = evaluate_predictions(labels, predictions, SCDB_LABEL_NAMES)
        report[name] = {
            "pickled_with_sklearn": pickled.pop() if pickled else sklearn.__version__,
            "hyperparameter": {"C": classifier.C} if hasattr(classifier, "C") else
                              {"alpha": classifier.alpha},
            "test_agreement_with_reported_predictions": float(np.mean(predictions == stored)),
            "demo_test_macro_f1": metrics["macro_f1"],
            "demo_test_accuracy": metrics["accuracy"],
        }

    out = ROOT / "results" / "demo_artifact_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
