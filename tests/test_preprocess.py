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
    text = "Decided April 4, 1955. (1982) No. 76-5663 348 U.S. 540 12th 1st opinion"
    normalized = normalize_text(text)
    assert not any(token.isdigit() for token in normalized.split())
    assert "12th" in normalized.split()
    assert "1st" in normalized.split()

    retained = normalize_text(text, drop_numeric_tokens=False)
    assert "1955" in retained
    assert "1982" in retained
    assert "76-5663" in retained
    assert "348" in retained and "540" in retained

    vectorizer = fit_on_train(
        build_tfidf_vectorizer(
            PreprocessingConfig(min_df=1, tfidf_ngram_range=(1, 2))
        ),
        [text, "opinion text"],
    )
    assert not any(key.isdigit() for key in vectorizer.vocabulary_)
    assert not any(
        any(part.isdigit() for part in key.split())
        for key in vectorizer.vocabulary_
    )


def test_transform_split_before_fit_raises():
    with pytest.raises(ValueError, match="fitted"):
        transform_split(build_tfidf_vectorizer(), ["new document"])


def test_fit_accepts_numpy_arrays():
    train = np.array(["one legal document", "another legal document"])
    vectorizer = fit_on_train(build_tfidf_vectorizer(), train)
    assert vectorizer.vocabulary_
