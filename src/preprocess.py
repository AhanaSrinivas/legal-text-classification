"""
Leakage-safe text normalization and sparse feature preparation.

Text normalization is deliberately conservative for legal opinions. In
particular, negations and modal/legal terms are not removed as stop words.
Vectorizers must be fitted with training text and only transformed on the
validation and test text.
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer


# These words can change the legal meaning of a sentence and must remain.
PRESERVED_LEGAL_TERMS = frozenset(
    {"not", "no", "nor", "never", "shall", "may", "might", "must", "against",
     "without", "except", "unless"}
)
LEGAL_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS.difference(PRESERVED_LEGAL_TERMS))
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without deleting legally meaningful words."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", normalized).strip().lower()


def normalize_texts(texts: Iterable[str]) -> List[str]:
    """Normalize a finite iterable of documents in input order."""
    return [normalize_text(text) for text in texts]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Memory-bounded defaults shared by the classical and topic pipelines."""

    tfidf_max_features: int = 50_000
    count_max_features: int = 30_000
    min_df: int = 2
    max_df: float = 0.98
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    count_ngram_range: tuple[int, int] = (1, 1)


def build_tfidf_vectorizer(
    config: PreprocessingConfig | None = None,
) -> TfidfVectorizer:
    """Create an unfitted TF-IDF vectorizer with legal-aware preprocessing."""
    config = config or PreprocessingConfig()
    return TfidfVectorizer(
        preprocessor=normalize_text,
        stop_words=sorted(LEGAL_STOP_WORDS),
        ngram_range=config.tfidf_ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.tfidf_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_count_vectorizer(
    config: PreprocessingConfig | None = None,
) -> CountVectorizer:
    """Create an unfitted count vectorizer for LDA/topic features."""
    config = config or PreprocessingConfig()
    return CountVectorizer(
        preprocessor=normalize_text,
        stop_words=sorted(LEGAL_STOP_WORDS),
        ngram_range=config.count_ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.count_max_features,
        dtype=np.int32,
    )


def fit_on_train(vectorizer, train_texts: Sequence[str]):
    """Fit a vectorizer only on train documents and return it.

    Callers should use ``transform_split`` for validation/test data; this
    function intentionally accepts only the training split by convention.
    """
    if not train_texts:
        raise ValueError("train_texts must contain at least one document")
    if len(train_texts) < 3:
        # Small fixtures cannot support corpus-level pruning thresholds.
        vectorizer.set_params(min_df=1, max_df=1.0)
    vectorizer.fit(train_texts)
    return vectorizer


def transform_split(vectorizer, texts: Sequence[str]):
    """Transform a split with an already train-fitted vectorizer."""
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("vectorizer must be fitted on the training split first")
    return vectorizer.transform(texts)
