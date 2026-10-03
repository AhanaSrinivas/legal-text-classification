import numpy as np
import pytest

from src.preprocess import (
    PreprocessingConfig,
    build_tfidf_vectorizer,
    fit_on_train,
    normalize_text,
    transform_split,
)


def test_legal_terms_survive_normalization_and_vectorization():
    terms = "not no nor never shall may might must against without except unless cannot none neither nothing nobody nowhere"
    assert all(term in normalize_text(terms).split() for term in terms.split())

    vectorizer = fit_on_train(
        build_tfidf_vectorizer(PreprocessingConfig(min_df=1)),
        [terms, "the court shall review the claim"],
    )
    for term in terms.split():
        assert term in vectorizer.vocabulary_


def test_numeric_tokens_are_dropped_by_default_and_can_be_retained():
    text = "348 1955 76-5663 opinion"
    assert "348" not in normalize_text(text).split()
    assert "1955" not in normalize_text(text).split()
    assert "348" in normalize_text(text, drop_numeric_tokens=False).split()

    vectorizer = fit_on_train(
        build_tfidf_vectorizer(PreprocessingConfig(min_df=1)),
        [text, "opinion text"],
    )
    assert "348" not in vectorizer.vocabulary_


def test_transform_split_before_fit_raises():
    with pytest.raises(ValueError, match="fitted"):
        transform_split(build_tfidf_vectorizer(), ["new document"])


def test_fit_accepts_numpy_arrays():
    train = np.array(["one legal document", "another legal document"])
    vectorizer = fit_on_train(build_tfidf_vectorizer(), train)
    assert vectorizer.vocabulary_
