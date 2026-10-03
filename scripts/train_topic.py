"""Train Phase 6 paper-method variants: LDA+LR and Doc2Vec+LR."""

import json
import os
import sys
from pathlib import Path

import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import SCDB_LABEL_NAMES, load_scotus_dataset
from src.evaluate import evaluate_predictions, metrics_row, save_confusion_matrix
from src.preprocess import (
    PreprocessingConfig,
    build_count_vectorizer,
    normalize_text,
    fit_on_train,
)


def main():
    dataset = load_scotus_dataset()
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]
    y_train = np.asarray(dataset["train"]["label"])
    y_val = np.asarray(dataset["validation"]["label"])
    y_test = np.asarray(dataset["test"]["label"])

    count_vectorizer = fit_on_train(
        build_count_vectorizer(
            PreprocessingConfig(count_max_features=500, min_df=5)
        ),
        train_texts,
    )
    x_train_counts = count_vectorizer.transform(train_texts)
    x_val_counts = count_vectorizer.transform(val_texts)
    x_test_counts = count_vectorizer.transform(test_texts)

    lda_candidates = {}
    for topics in (10, 20, 30, 40):
        lda = LatentDirichletAllocation(
            n_components=topics,
            max_iter=2,
            learning_method="batch",
            random_state=42,
            n_jobs=1,
        )
        train_topics = lda.fit_transform(x_train_counts)
        val_topics = lda.transform(x_val_counts)
        lda_model = LogisticRegression(
            C=1.0, max_iter=100, solver="lbfgs", random_state=42
        ).fit(train_topics, y_train)
        val_metrics = evaluate_predictions(
            y_val, lda_model.predict(val_topics), SCDB_LABEL_NAMES
        )
        lda_candidates[topics] = {
            "lda": lda,
            "model": lda_model,
            "validation": val_metrics,
        }

    best_topics = max(
        lda_candidates, key=lambda topics: lda_candidates[topics]["validation"]["macro_f1"]
    )
    best_lda = lda_candidates[best_topics]
    test_metrics = evaluate_predictions(
        y_test,
        best_lda["model"].predict(best_lda["lda"].transform(x_test_counts)),
        SCDB_LABEL_NAMES,
    )
    lda_reports = {
        str(topics): candidate["validation"] for topics, candidate in lda_candidates.items()
    }
    lda_reports[str(best_topics)]["test"] = test_metrics

    # Doc2Vec is trained only on train documents; validation and test use infer_vector.
    train_documents = [
        TaggedDocument(normalize_text(text).split()[:500], [index])
        for index, text in enumerate(train_texts)
    ]
    doc2vec = Doc2Vec(
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        epochs=5,
        dm=1,
        seed=42,
    )
    doc2vec.build_vocab(train_documents)
    doc2vec.train(
        train_documents,
        total_examples=doc2vec.corpus_count,
        epochs=doc2vec.epochs,
    )

    def infer(texts):
        return np.vstack([
            doc2vec.infer_vector(normalize_text(text).split()[:500], epochs=5)
            for text in texts
        ])

    doc_train = np.vstack([doc2vec.dv[index] for index in range(len(train_texts))])
    doc_val = infer(val_texts)
    doc_test = infer(test_texts)
    doc2vec_lr = LogisticRegression(
        C=1.0, max_iter=100, solver="lbfgs", random_state=42
    ).fit(doc_train, y_train)
    doc_val_metrics = evaluate_predictions(
        y_val, doc2vec_lr.predict(doc_val), SCDB_LABEL_NAMES
    )
    doc_test_metrics = evaluate_predictions(
        y_test, doc2vec_lr.predict(doc_test), SCDB_LABEL_NAMES
    )

    metrics_path = Path("results/metrics.csv")
    metrics = pd.read_csv(metrics_path)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame([
                metrics_row("lda_lr", "validation", best_lda["validation"]),
                metrics_row("lda_lr", "test", test_metrics),
                metrics_row("doc2vec_lr", "validation", doc_val_metrics),
                metrics_row("doc2vec_lr", "test", doc_test_metrics),
            ]),
        ],
        ignore_index=True,
    )
    metrics.to_csv(metrics_path, index=False)

    reports_path = Path("results/classification_reports/classical_reports.json")
    reports = json.loads(reports_path.read_text())
    reports["lda_lr"] = {
        "selected_topics": best_topics,
        "topic_sweep_validation": lda_reports,
        "test": test_metrics,
    }
    reports["doc2vec_lr"] = {
        "test": doc_test_metrics,
        "validation": doc_val_metrics,
    }
    reports_path.write_text(json.dumps(reports, indent=2))
    Path("results/topic_selection.json").write_text(json.dumps({
        "lda_input": "raw count frequencies",
        "topics_considered": [10, 20, 30, 40],
        "selected_topics": best_topics,
        "validation": lda_reports,
        "doc2vec": {
            "vector_size": 100,
            "window": 5,
            "epochs": 5,
            "max_tokens_per_document": 500,
            "trained_on": "train only",
        },
    }, indent=2))
    save_confusion_matrix(
        test_metrics, SCDB_LABEL_NAMES, "results/figures/confusion_matrix_lda_lr.png"
    )
    save_confusion_matrix(
        doc_test_metrics, SCDB_LABEL_NAMES,
        "results/figures/confusion_matrix_doc2vec_lr.png",
    )
    print(metrics.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
