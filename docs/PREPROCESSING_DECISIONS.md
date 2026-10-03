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
  `without`, `except`, `unless`, `cannot`, `none`, `neither`, `nothing`,
  `nobody`, and `nowhere`. These terms can change the meaning of a legal
  statement. The final six terms were verified to be present in sklearn's
  `ENGLISH_STOP_WORDS` and are explicitly removed from that list.
- The following sklearn stop words remain removed: `under`, `upon`, `within`,
  `whether`, `between`, `before`, `after`, `hereby`, `herein`, `therefore`,
  and `whereas`. In this baseline they function primarily as general
  relational, temporal, or discourse words; removing them limits vocabulary
  size without asserting an accuracy benefit. `thereof` is not in sklearn's
  default stop-word list and is therefore retained.
- Pure numeric whitespace-delimited tokens are dropped by default
  (`drop_numeric_tokens=True`). Reporter citations, docket numbers, and years
  can encode time-period fingerprints that do not transfer from the
  chronological training period to the test period. Set the option to
  `False` to retain them for an explicit ablation. No accuracy improvement is
  claimed until measured on validation.
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
metrics will be recorded under `results/` in later phases. The full-train
TF-IDF (unigram + bigram) fit is benchmarked by
`scripts/benchmark_preprocessing.py`, which records wall time and peak
process RSS in `results/preprocessing_benchmark.json`. If either the
10-minute or approximately 4 GB threshold is exceeded, the script records
and uses a unigram fallback.

The executed benchmark retained the requested unigram+bigram configuration:
263.132 seconds wall time, 1.541 GB peak process RSS, and a 50,000-feature
vocabulary. These are preprocessing resource measurements, not model accuracy
results.

## Reproducibility

Preprocessing itself has no random operation. Model and split-level seeds are
set to 42 by the training scripts that consume these features.
