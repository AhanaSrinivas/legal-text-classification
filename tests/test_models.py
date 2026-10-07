"""Small-sample fits of the classical pipelines (shapes and label handling)."""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from src.preprocess import (
    PreprocessingConfig,
    build_count_vectorizer,
    build_tfidf_vectorizer,
    fit_on_train,
    transform_split,
)


TRAIN = [
    "the defendant was convicted and the search warrant was invalid",
    "police search without a warrant violated the fourth amendment",
    "the tax court assessed a federal income tax deficiency",
    "the commissioner of internal revenue disallowed the tax deduction",
    "the union and the employer bargained under the labor act",
    "the labor board found an unfair labor practice by the union",
]
LABELS = np.array([0, 0, 11, 11, 6, 6])
TEST = ["a warrantless search by police", "income tax deduction disallowed"]


def _features():
    vectorizer = fit_on_train(
        build_tfidf_vectorizer(PreprocessingConfig(min_df=1)), TRAIN
    )
    return vectorizer, vectorizer.transform(TRAIN), transform_split(vectorizer, TEST)


def test_tfidf_linear_svc_fits_and_predicts_known_labels():
    _, x_train, x_test = _features()
    model = LinearSVC(C=1.0, loss="squared_hinge", random_state=42).fit(x_train, LABELS)
    assert model.decision_function(x_test).shape == (2, 3)
    assert list(model.predict(x_test)) == [0, 11]


def test_tfidf_sgd_log_loss_gives_probabilities():
    _, x_train, x_test = _features()
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                          max_iter=50, tol=1e-3, random_state=42).fit(x_train, LABELS)
    probabilities = model.predict_proba(x_test)
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_tfidf_features_are_l2_normalised_float32():
    _, x_train, _ = _features()
    assert x_train.dtype == np.float32
    norms = np.sqrt(np.asarray(x_train.multiply(x_train).sum(axis=1))).ravel()
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_count_vectorizer_gives_integer_counts_for_lda():
    vectorizer = fit_on_train(build_count_vectorizer(PreprocessingConfig(min_df=1)), TRAIN)
    counts = vectorizer.transform(TRAIN)
    assert counts.dtype == np.int32
    assert counts.min() >= 0
    assert all(" " not in term for term in vectorizer.vocabulary_)  # unigrams only
