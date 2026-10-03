"""Build only the CPU demo artifacts; never modify results or evaluation outputs."""

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_scotus_dataset
from src.preprocess import PreprocessingConfig, build_tfidf_vectorizer, fit_on_train


def main():
    started = time.perf_counter()
    output_dir = Path("models/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_scotus_dataset()
    config = PreprocessingConfig(tfidf_ngram_range=(1, 2), tfidf_max_features=10_000)
    vectorizer = fit_on_train(build_tfidf_vectorizer(config), dataset["train"]["text"])
    x_train = vectorizer.transform(dataset["train"]["text"])
    y_train = np.asarray(dataset["train"]["label"])

    svc = LinearSVC(C=0.1, loss="squared_hinge", random_state=42).fit(x_train, y_train)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(svc, output_dir / "tfidf_linear_svc.joblib")

    try:
        lr = LogisticRegression(
            C=1.0, class_weight=None, max_iter=5, solver="saga",
            tol=0.5, random_state=42,
        ).fit(x_train, y_train)
        joblib.dump(lr, output_dir / "tfidf_logistic_regression.joblib")
        print("Created TF-IDF + LinearSVC and TF-IDF + LogisticRegression demo artifacts.")
    except Exception as exc:
        print(f"Logistic Regression demo artifact was not created: {exc}", file=sys.stderr)
        print("The required LinearSVC demo artifact was created.", file=sys.stderr)
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
