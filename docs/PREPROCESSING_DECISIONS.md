# Preprocessing Decisions

## Scope

Phase 4 provides reusable text normalization and sparse-vector preparation in
[`src/preprocess.py`](../src/preprocess.py). The implementation is shared by
the planned classical and topic-model pipelines.

## Legal-text normalization

- Unicode is normalized with NFKC.
- Whitespace is collapsed and text is lower-cased.
- Punctuation, citations, and short documents are not removed by a custom
  legal-specific filter.
- The stop-word list starts from scikit-learn's English list, but preserves
  `not`, `no`, `nor`, `never`, `shall`, `may`, `might`, `must`, `against`,
  `without`, `except`, and `unless`. These terms can change the meaning of a
  legal statement.
- No stemming or lemmatization is applied. This avoids changing legal terms
  before a baseline has been measured.

## Leakage prevention

Vectorizer vocabulary, document-frequency statistics, and IDF values are fit
on the training split only. Validation and test documents are passed through
`transform_split`; they are never passed to `fit` or `fit_transform`.
Downstream topic models and Doc2Vec models must follow the same rule.

## Feature limits

The default TF-IDF vocabulary is capped at 50,000 features and the count
vocabulary used for LDA is capped at 30,000 features. Both use `min_df=2` and
`max_df=0.98` to limit memory use on the CPU environment. These are
engineering defaults, not measured model results; final model settings and
metrics will be recorded under `results/` in later phases.

## Reproducibility

Preprocessing itself has no random operation. Model and split-level seeds are
set to 42 by the training scripts that consume these features.
