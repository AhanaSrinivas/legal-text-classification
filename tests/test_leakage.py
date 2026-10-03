import pytest

from src.preprocess import build_tfidf_vectorizer, fit_on_train, transform_split


def test_validation_only_token_is_absent_from_train_vocabulary():
    vectorizer = fit_on_train(
        build_tfidf_vectorizer(),
        ["train alpha document", "train beta document"],
    )
    assert "validationonlytoken" not in vectorizer.vocabulary_
    transformed = transform_split(vectorizer, ["validationonlytoken"])
    assert transformed.shape == (1, len(vectorizer.vocabulary_))
    assert transformed.nnz == 0


def test_empty_train_split_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        fit_on_train(build_tfidf_vectorizer(), [])
