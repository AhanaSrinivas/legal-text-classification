# Classification of Legal Text: US Supreme Court Opinions

**Course:** UE24CS352A Machine Learning mini-project
**Task:** predict the SCDB issue area of a US Supreme Court opinion
**Dataset:** LexGLUE SCOTUS (`coastalcph/lex_glue`, config `scotus`)

<!-- BEGIN GENERATED: summary -->
Best model on the test split: **TF-IDF + LinearSVC**, macro F1 0.635 (95% bootstrap CI 0.598–0.691), accuracy 0.737. Fine-tuned Legal-BERT on the first 512 tokens reaches macro F1 0.512 (CI 0.476–0.556); the paper's LDA and Doc2Vec pipelines (reduced CPU configurations) reach 0.321 and 0.322.
<!-- END GENERATED: summary -->

Numbers in this README are generated from `results/` by
`scripts/make_readme_results.py`; do not edit the generated sections by hand.

## 1. Problem statement

Given the full text of a US Supreme Court opinion, assign it to one of the
Supreme Court Database (SCDB) issue areas, such as Criminal Procedure, Civil
Rights, First Amendment or Economic Activity. This is single-label,
multi-class document classification with a strongly imbalanced label
distribution and very long documents.

## 2. Motivation

Every opinion has to be catalogued by subject before it can be found in
case-law research, and doing this by hand is slow. Legal opinions are hard
for standard NLP: they are long, full of citations and procedural
boilerplate, and use domain vocabulary. The reference project (Iyer, Stanford
CS229, 2020) tried LDA and Doc2Vec features with logistic regression and could
not run BERT because it ran out of memory. We reproduce those two methods on a
public benchmark with fixed splits, add strong TF-IDF baselines, and fine-tune
a legal-domain Transformer, reporting every result honestly, including the
weak ones.

## 3. Dataset

<!-- BEGIN GENERATED: dataset -->
- **Source:** LexGLUE benchmark, `coastalcph/lex_glue`, config `scotus` on the Hugging Face Hub, loaded with `datasets` 5.0.0. Labels are SCDB issue areas (Supreme Court Database, Spaeth et al.).
- **Size and split (official):** 7,800 opinions: train 5,000, validation 1,400, test 1,400. LexGLUE splits SCOTUS chronologically: train 1946–1982, validation 1982–1991, test 1991–2016 (Chalkidis et al., 2022, Section 3).
- **Columns:** `text` (full opinion) and `label` (13 classes). SCDB defines one more issue area (Private Action) that has no cases in this corpus.
- **Quality checks:** 0 empty texts; 0 exact duplicates within splits and 0 across splits (SHA-256 of each text).
- **Imbalance:** largest training class Economic Activity (1,043 documents), smallest Miscellaneous (3), ratio 347.7:1. Rarest test classes: Miscellaneous (3), Interstate Relations (15), Attorneys (17).
- **Length:** median 4,364.5 words (train), 7,591 (validation), 6,883 (test). On a 500-document train sample, the Legal-BERT tokenizer gives a median of 5,470.5 tokens, 93% of documents exceed 512 tokens, and the first 512 tokens cover about 20% of a document on average.

Sources: `results/dataset_info.json`, `results/eda_stats.json`.
<!-- END GENERATED: dataset -->

The dataset is downloaded automatically by `src/data.py` from the Hugging Face
Hub on first use (no account needed). Why we chose it over the paper's
`textacy` scrape is recorded in `docs/DATASET_DECISION.md`.

## 4. Methodology

1. **Split discipline.** We use the official train/validation/test splits.
   Every hyperparameter is chosen on validation; final models are fitted on
   train only; the test split is predicted once per final model.
2. **No leakage.** Vectorizers, LDA and Doc2Vec are fitted on the training
   split only and only `transform`/`infer_vector` is applied to validation and
   test (`src/preprocess.py`, `tests/test_leakage.py`).
3. **Primary metric: macro F1**, because the classes are very imbalanced and
   macro F1 weights every issue area equally. Accuracy and weighted F1 are
   reported alongside, and every per-class table carries its support.
4. **Uncertainty.** Percentile bootstrap confidence intervals for test macro F1.
5. **Error analysis** from saved predictions: confused class pairs, accuracy by
   document-length quartile, and concrete misclassified cases
   (`results/error_analysis.md`).

## 5. Preprocessing

Implemented in `src/preprocess.py`; decisions and reasons are in
`docs/PREPROCESSING_DECISIONS.md`.

- Unicode NFKC normalisation, lower-casing, whitespace collapsing.
- Standalone numbers are removed (reporter citations, docket numbers, years),
  because they encode the time period rather than the topic. Tokens such as
  `12th` or `1st` are kept.
- scikit-learn's English stop-word list **minus** words that change legal
  meaning: `not, no, nor, never, shall, may, might, must, against, without,
  except, unless, cannot, none, neither, nothing, nobody, nowhere`.
- No stemming or lemmatisation.
- TF-IDF: word unigrams and bigrams, `min_df=2`, `max_df=0.98`, sublinear term
  frequency, float32.

## 6. Models

<!-- BEGIN GENERATED: models -->
| Model | Features | Classifier and selected setting | Selection evidence |
|---|---|---|---|
| Majority baseline | none | always predicts the training majority class (Economic Activity) | `results/majority_baseline.json` |
| TF-IDF + LinearSVC | TF-IDF, unigrams + bigrams, 10,000 features | `LinearSVC` (L2, squared hinge), C = 1.0 chosen from 0.01, 0.1, 1.0 | `results/classical_selection.json` |
| TF-IDF + LR (final) | same TF-IDF | `SGDClassifier(loss="log_loss", penalty="l2")`, alpha = 1e-5 chosen from 1e-6, 1e-5, 1e-4; converged in 9 of 50 epochs | `results/converged_lr_selection.json` |
| TF-IDF + LR (saga) | same TF-IDF | `LogisticRegression(solver="saga")`, max_iter = 5, tol = 0.5; **under-converged**, kept only for transparency | `results/classical_selection.json` |
| LDA + LR (paper method) | LDA topic mixture on raw counts (500 terms, min_df 5, 2 batch iterations) | `LogisticRegression` (lbfgs); K = 40 topics chosen from 10, 20, 30, 40 | `results/topic_selection.json` |
| Doc2Vec + LR (paper method) | PV-DM, 100 dims, window 5, 5 epochs, first 500 tokens per document; trained on train only, vectors inferred for val/test | `LogisticRegression` (lbfgs) | `results/topic_selection.json` |
| Legal-BERT | `nlpaueb/legal-bert-base-uncased`, first 512 tokens | fine-tuned 3 epochs, batch 8 x accumulation 2 = 16, learning rate 2e-5, weight decay 0.01, linear warm-up over 10% of steps, mixed precision, seed 42, plain cross-entropy (no class weights), best epoch by validation macro F1; Tesla T4, 19.6 minutes | `results/transformer_run_info.json` |

All hyperparameters were chosen on the validation split; final models were fitted on the training split only.
<!-- END GENERATED: models -->

Regularisation is L2 everywhere (and weight decay for Legal-BERT); no L1/Lasso
and no feature selection beyond the vocabulary cap.

## 7. Evaluation

`scripts/evaluate_all.py` rebuilds all evaluation outputs from the saved
prediction archives in `results/predictions/`, so no model is retrained:
`results/metrics.csv`, `results/per_class_f1.csv`, `results/bootstrap_ci.json`,
the confusion-matrix figures, `results/figures/model_comparison.png` and
`results/error_analysis.md`. `tests/test_results_integrity.py` recomputes
`metrics.csv` and `per_class_f1.csv` from the archives on every test run.

## 8. Results

<!-- BEGIN GENERATED: results -->
Official LexGLUE test split (used once per model) and validation split. Macro F1 is the primary metric.

| Model | Test macro F1 | 95% CI (macro F1) | Test accuracy | Test weighted F1 | Val macro F1 | Val accuracy | Val weighted F1 |
|---|---:|:---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 0.635 | [0.598, 0.691] | 0.737 | 0.723 | 0.724 | 0.789 | 0.781 |
| TF-IDF + LR (SGD, log loss) | 0.629 | [0.590, 0.682] | 0.732 | 0.719 | 0.712 | 0.786 | 0.779 |
| Legal-BERT (first 512 tokens) | 0.512 | [0.476, 0.556] | 0.724 | 0.694 | 0.594 | 0.738 | 0.714 |
| LDA topics + LR | 0.321 | [0.298, 0.350] | 0.546 | 0.514 | 0.349 | 0.569 | 0.533 |
| Doc2Vec + LR | 0.322 | [0.292, 0.353] | 0.474 | 0.456 | 0.406 | 0.529 | 0.508 |
| TF-IDF + LR (saga, under-converged) | 0.470 | [0.434, 0.510] | 0.680 | 0.652 | 0.580 | 0.721 | 0.700 |
| Majority class (train) | 0.024 | [0.022, 0.027] | 0.186 | 0.058 | 0.021 | 0.161 | 0.045 |

CIs are percentile bootstrap intervals over 1,000 resamples of the test set (seed 42). They are unpaired and capture test-set sampling only, not training randomness. The LinearSVC and Legal-BERT intervals do not overlap.

**Legal-BERT run-to-run variation.** Three full runs with identical settings gave test macro F1 0.5171 (first run; its files were overwritten, value recorded in `docs/RUN_LOG.md`), 0.5124 (the reported run: `reports/colab_run_log.ipynb`, predictions in `results/transformer_predictions.npz`) and 0.5319 (rerun: `results/transformer_run2_info.json`). We report the 0.5124 run and treat the range 0.5124–0.5319 as run-to-run variation; no best-run selection was made. The demo weights come from the rerun.

**Per-class highlights (test).** Legal-BERT is strong on Criminal Procedure (F1 0.866), Federal Taxation (0.835) and First Amendment (0.804), but scores F1 = 0 on Attorneys and Miscellaneous and recalls only 7% of Federalism and 13% of Interstate Relations cases. LinearSVC's weakest classes are Miscellaneous (F1 0.000, test support 3) and Federalism (0.336). Most frequent confusion (count in brackets): Federalism → Economic Activity (37) for LinearSVC and Federalism → Economic Activity (49) for Legal-BERT.

Figures: `results/figures/model_comparison.png`, `results/figures/confusion_matrix_normalized_*.png`; per-class scores with support: `results/per_class_f1.csv`; error analysis: `results/error_analysis.md`.
<!-- END GENERATED: results -->

**Reading the results.** Linear models on the full TF-IDF document beat
Legal-BERT that sees only the opening of each opinion. Our hypotheses, none of
them tested: (1) truncation, since the opening is mostly the caption,
citations and counsel names; (2) no class weighting, which hurts the rare
classes Legal-BERT never predicts; (3) a single short fine-tuning run. The two
paper methods do worst, but in reduced configurations, so this does not show
that LDA or Doc2Vec are weak in general. The reference paper's own numbers
were not used as targets: its dataset and split differ, its Table 1 column
labels contradict its text, and it reports no BERT result
(`docs/reference_notes.md`).

## 9. Installation

Python 3.13 was used (`requirements.txt` pins every package).

macOS / Linux:

```bash
git clone https://github.com/AhanaSrinivas/legal-text-classification.git
cd legal-text-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/AhanaSrinivas/legal-text-classification.git
cd legal-text-classification
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Model artifacts (needed only for live demo predictions)

Fitted models are not stored in Git. They are on the Hugging Face Hub in
`AhanaSrinivas/legal-text-classification-artifacts` (public):

| Hugging Face path | Local path | Used for |
|---|---|---|
| `sklearn_artifacts/demo/tfidf_vectorizer.joblib` | `models/demo/tfidf_vectorizer.joblib` | TF-IDF features |
| `sklearn_artifacts/demo/tfidf_linear_svc.joblib` | `models/demo/tfidf_linear_svc.joblib` | live LinearSVC |
| `sklearn_artifacts/demo/tfidf_logistic_regression.joblib` | `models/demo/tfidf_logistic_regression.joblib` | live LR |
| `legal_bert_state.pt` (438 MB) | `models/legal_bert_state.pt` | live Legal-BERT |

`scripts/download_artifacts.py` downloads all four files. The current version
reads a Hugging Face access token from the `HF_TOKEN` environment variable
(any free read token works because the repository is public) and never writes
it to disk:

```bash
HF_TOKEN=<your-read-token> python scripts/download_artifacts.py      # macOS / Linux
```

```powershell
$env:HF_TOKEN="<your-read-token>"; python scripts\download_artifacts.py   # Windows
```

Without a token the script exits cleanly and downloads nothing; the files can
also be downloaded from the repository's "Files" page into the local paths
above. Check what you have with `python scripts/check_demo_artifacts.py`.

## 10. Exact commands

Run from the repository root with the virtual environment active.

| Purpose | Command | Needs |
|---|---|---|
| Run the test suite | `python -m pytest tests -q` | nothing extra (artifact tests skip if files are absent) |
| Also test live Legal-BERT | `LTC_RUN_BERT=1 python -m pytest tests -q` | `models/legal_bert_state.pt`, internet once for the tokenizer |
| Rebuild evaluation outputs from saved predictions | `python scripts/evaluate_all.py` | dataset download |
| Compare demo artifacts with reported predictions | `python scripts/check_demo_artifacts.py` | artifacts, dataset |
| Rebuild README tables, write-up PDF, slides, viva notes and run the number audit | `python run_all.py` | nothing extra |
| Start the demo | `streamlit run app.py` | artifacts for live predictions |

Training scripts (not needed to reproduce the reported numbers; they retrain
and overwrite results): `scripts/01_run_eda.py`, `scripts/benchmark_preprocessing.py`,
`scripts/train_classical.py`, `scripts/train_converged_lr.py`,
`scripts/train_topic.py`, and `scripts/04_train_transformer.py` (GPU; see
`docs/COLAB_RUNBOOK.md`). Legal-BERT was trained on Google Colab; the run is
logged in `reports/colab_run_log.ipynb`.

## 11. Streamlit demo

```bash
streamlit run app.py
```

Then open http://localhost:8501.

- **Classify** tab: pick one of the committed sample test cases (one per
  class, `data/sample_cases.json`, first 3,000 characters of each opinion) or
  paste your own text, choose models, and press *Classify*. Each model shows
  its prediction and top-5 scores (probabilities for LR and Legal-BERT, margin
  scores for LinearSVC). Legal-BERT shows how much of the text it could read.
  For sample cases, the stored predictions that every model made on the full
  opinion are shown below for comparison.
- **Results** tab: test and validation metrics with CIs, confusion matrices and
  per-class scores with support, all read from `results/`.
- **About & limitations** tab.

Without artifacts the app still starts, marks the missing models and serves
the Results tab. Legal-BERT on CPU takes a few seconds per document; the
first use downloads its tokenizer and config (small) from the Hugging Face
Hub, so run it once with internet before a live demo. The live Legal-BERT
weights come from a rerun with identical settings, so its predictions can
differ from the reported run's stored predictions.

## 12. Repository structure

```
legal-text-classification/
├── README.md, PROJECT_PLAN.md, requirements.txt, pytest.ini, .gitignore
├── app.py                         Streamlit demo
├── run_all.py                     rebuild documents from results/ and audit them (PASS/FAIL)
├── src/
│   ├── data.py                    dataset loading, schema and duplicate checks, label names
│   ├── preprocess.py              normalisation, legal stop words, train-only vectorizers
│   ├── evaluate.py                shared metrics, per-class reports, confusion matrices
│   ├── utils.py                   seeds, hardware detection, JSON helpers
│   ├── predict.py                 optional artifact download from Hugging Face
│   ├── demo.py                    demo helpers (loading, prediction, stored results)
│   └── report_facts.py            every number used in the documents, read from results/
├── scripts/
│   ├── 01_run_eda.py              dataset validation and EDA
│   ├── benchmark_preprocessing.py TF-IDF fit time and memory benchmark
│   ├── train_classical.py         TF-IDF + LinearSVC and saga LR
│   ├── train_converged_lr.py      TF-IDF + SGD logistic regression (final LR)
│   ├── train_topic.py             LDA + LR and Doc2Vec + LR
│   ├── 04_train_transformer.py    Legal-BERT fine-tuning (GPU)
│   ├── evaluate_all.py            evaluation outputs from saved predictions
│   ├── make_demo_artifacts.py     CPU rebuild of the demo TF-IDF models
│   ├── download_artifacts.py      fetch demo artifacts from Hugging Face
│   ├── upload_artifacts.py        upload artifacts (repository owner only)
│   ├── check_demo_artifacts.py    compare demo artifacts with reported predictions
│   ├── make_readme_results.py     generated README sections
│   ├── make_slides.py             reports/slides.pptx
│   ├── make_viva_qa.py            reports/VIVA_QA.md
│   └── audit_numbers.py           checks every number in the documents against results/
├── reports/
│   ├── writeup.py / writeup.pdf   two-page write-up (source and PDF)
│   ├── slides.pptx                review deck with speaker notes
│   ├── VIVA_QA.md                 viva preparation (generated from templates/VIVA_QA.template.md)
│   ├── CONTRIBUTIONS.md           team contributions and AI-assistance note
│   ├── colab_run_log.ipynb        reported Legal-BERT run
│   └── colab_run2_demo_weights.ipynb  rerun that produced the demo weights
├── results/                       metrics, predictions, CIs, figures, error analysis, test report
├── data/sample_cases.json         one test case per class for the demo and tests
├── models/README.md               artifact notes (artifacts themselves are not in Git)
├── docs/                          guidelines, handoff, run log, decisions, citations, final validation
└── tests/                         pytest suite
```

## 13. Limitations

<!-- BEGIN GENERATED: limitations -->
- **Truncation.** Legal-BERT reads only the first 512 tokens, while the median sampled document has about 5,470 tokens; the first 512 tokens cover about 20% of a document on average. Its result is a result for truncated input, not for Legal-BERT on full opinions.
- **Reduced paper methods.** LDA (500 count features, 2 iterations) and Doc2Vec (5 epochs, first 500 tokens) ran in reduced CPU configurations. The selected K = 40 is at the edge of the grid. Their scores are not a fair measure of what these methods can do.
- **One seed.** Each model was trained once (seed 42). Legal-BERT varied from 0.5124 to 0.5319 test macro F1 across three identical runs.
- **Chronological split.** Test cases (1991–2016) come from a later period than training cases (1946–1982). Every trained model drops from validation to test (LinearSVC by 0.089 macro F1); temporal drift is a plausible cause but was not tested.
- **Tiny rare classes.** Test support is Miscellaneous (3), Interstate Relations (15), Attorneys (17), so one prediction can move a class F1 a lot and macro F1 is noisy.
- **Unpaired CIs.** Bootstrap intervals are per model and capture test sampling only; they are not a paired significance test.
- **No class weighting** in any model, and no hyperparameter search beyond the small grids above.
- **Demo artifacts.** The demo LinearSVC artifact uses C = 0.1, not the reported C = 1.0; on the test set it agrees with the reported predictions on 88% of documents (macro F1 0.505). The demo LR agrees on 97%. The `.joblib` files were saved with scikit-learn 1.9.1, which differs from the pinned version. Source: `results/demo_artifact_check.json`.
<!-- END GENERATED: limitations -->

## 14. Future work

- Long-document models: hierarchical Legal-BERT over paragraphs (as LexGLUE
  does for SCOTUS) or Longformer/BigBird, instead of reading only the opening.
- Class-weighted or focal loss and several seeds per model, with paired
  bootstrap or McNemar tests between models.
- LDA and Doc2Vec in full configurations (larger vocabulary, more iterations
  and epochs, whole documents) and a wider K grid.
- L1 or elastic-net regularisation for sparse, interpretable TF-IDF models.
- Removing the caption/counsel header before classification.

## 15. References

1. K. Iyer. *Classification of Legal Text.* Stanford CS229 project report,
   Spring 2020. Local copy: `docs/references/Iyer.pdf`.
2. I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos,
   D. M. Katz, N. Aletras. *LexGLUE: A Benchmark Dataset for Legal Language
   Understanding in English.* Proceedings of ACL 2022. arXiv:2110.00976.
3. H. J. Spaeth, L. Epstein, J. A. Segal, A. D. Martin, T. J. Ruger,
   S. C. Benesh. *Supreme Court Database*, Version 2020 Release 01.
   Washington University Law, 2020. http://scdb.wustl.edu.
4. I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras,
   I. Androutsopoulos. *LEGAL-BERT: The Muppets straight out of Law School.*
   Findings of EMNLP 2020 (model `nlpaueb/legal-bert-base-uncased`).

How each citation was checked is recorded in `docs/CITATIONS.md`.

## 16. Team and use of AI assistance

Team members and the split of work are in `reports/CONTRIBUTIONS.md`; the Git
history shows each member's commits.

**Use of AI assistance.** AI coding assistants were used during the project
to draft code, tests and documentation. Commits made with an assistant are
marked with a `Co-Authored-By` trailer. Every number in the documents is
generated from files in `results/` and checked by `scripts/audit_numbers.py`;
the team reviewed the generated code and text and is responsible for it.
