# Classification of Legal Text (US Supreme Court Opinions)
**Course**: UE24CS352A - Machine Learning Mini-Project  
**Domain**: Legal Natural Language Processing (NLP)  
**Dataset**: LexGLUE SCOTUS Benchmark (`coastalcph/lex_glue`, `scotus` config)  

---

## 1. Project Overview & Motivation
Classifying legal texts is a core challenge in legal informatics. Legal opinions are characterized by long-running syntactic constructions, extensive citations to judicial precedent, and non-standard domain jargon. Automated categorization of court decisions into human-curated taxonomy areas (such as the Supreme Court Database issue areas) enables streamlined case-law research, assists public defenders, and democratizes access to justice.

This project reproduces and extends the methodology investigated by Krithika Iyer (Stanford CS229, 2020, *Classification of Legal Text*):
- **Paper Methods**: Evaluates Latent Dirichlet Allocation (LDA) topic mixtures + Logistic Regression, and Paragraph Vector (Doc2Vec) dense embeddings + Logistic Regression.
- **Our Baselines & Extensions**: Implements TF-IDF classical ML baselines (SGD logistic-loss classifier and LinearSVC), resolves data leakage via strict split isolation, evaluates a domain-adapted Transformer (`nlpaueb/legal-bert-base-uncased`), and provides honest empirical reporting on long-document truncation (512 tokens). The Streamlit application remains planned for Phase 9.

---

## 2. Paper Method vs. Our Project Strategy

| Component | Reference Paper (Iyer 2020) | Our Implementation & Reproduction | Status / Notes |
|:---|:---|:---|:---|
| **Dataset Source** | `textacy` scrape (~8,200 opinions) | LexGLUE SCOTUS (`coastalcph/lex_glue`) | Verified public benchmark, non-leaking splits |
| **Splitting Method** | Random split; details not specified in the paper | Chronological split recorded by LexGLUE metadata (boundary years are source-dataset metadata; verify against `results/dataset_info.json`) | Temporal ordering is documented; no causal leakage claim |
| **Class Taxonomy** | 15 issue areas / 279 issue codes | 13 active SCDB issue areas (0-12) | Class 14 ('Private Action') is absent from the adopted LexGLUE SCOTUS corpus; see `docs/DATASET_DECISION.md` |
| **Classical Baselines**| None | TF-IDF + Logistic Regression, TF-IDF + LinearSVC | Essential classical benchmarks |
| **Topic Modeling** | LDA (TF-IDF input) + Logistic Regression | LDA (Count input, topic sweep $K \in \{10,20,30,40\}$) + LR | Primary input uses count frequencies; TF-IDF difference noted |
| **Document Vectors** | Doc2Vec + Logistic Regression | Doc2Vec (PV-DBOW / PV-DM, fit strictly on train) + LR | Exact inference hygiene on val/test |
| **Transformer** | Failed (Ran out of memory on Colab; no results) | GPU-trained `nlpaueb/legal-bert-base-uncased` | Implemented GPU run; metadata and predictions are in `results/` |
| **Interactive Demo**| None | Streamlit Web Application (`app.py`) | Planned for Phase 9; demo artifacts can be generated locally |
| **Test Suite** | None | Pytest unit test suite (`tests/`) | Validates data hygiene, preprocessing, leakage, and models |

---

## 3. Verified Dataset Facts
From Phase 3 Exploratory Data Analysis (`results/dataset_info.json`, `results/eda_stats.json`):
- **Corpus Size**: 7,800 opinions across 3 chronological splits:
  - **Train**: 5,000 cases (1946–1982)
  - **Validation**: 1,400 cases (1982–1991)
  - **Test**: 1,400 cases (1991–2016)
- **Target Classes**: 13 SCDB Issue Areas:
  `Criminal Procedure`, `Civil Rights`, `First Amendment`, `Due Process`, `Privacy`, `Attorneys`, `Unions`, `Economic Activity`, `Judicial Power`, `Federalism`, `Interstate Relations`, `Federal Taxation`, `Miscellaneous`.
- **Class Imbalance**: Imbalance ratio of 347.7:1 between Economic Activity (1,043 train docs) and Miscellaneous (3 train docs).
- **Primary Metric**: **Macro-averaged F1 Score** (Majority-class test baseline achieves 18.57% accuracy and 0.0241 Macro F1).
- **Subword Token Lengths** (`legal-bert-base-uncased`): Median 5,470.5 tokens; 93.0% of documents exceed 512 tokens in the measured 500-document train sample (see `results/eda_stats.json`).

---

## 4. Repository Structure
```
legal-text-classification/
├── README.md                 # Primary project overview and instructions
├── PROJECT_PLAN.md           # System architecture, pipeline, and evaluation design
├── requirements.txt          # Python dependencies
├── .gitignore                # Filters caches, checkpoints, and spreadsheets
├── src/                      # Reusable modular source code
│   ├── data.py               # Dataset loading, schema validation, duplicate audit
│   ├── preprocess.py         # Legal text normalization and train-only vectorizer fitting
│   ├── evaluate.py           # Shared metrics, reports, and confusion matrices
│   └── utils.py              # Hardware detection, seeds, and metric I/O
├── scripts/                  # Standalone execution stages
│   ├── 01_run_eda.py         # Dataset validation & EDA report
│   └── 04_train_transformer.py # Legal-BERT training script (resumable)
├── data/
│   └── sample_cases.json     # Seeded class-stratified test sample (45 KB)
├── results/                  # Real experimental outputs (metrics, JSON, figures)
│   ├── dataset_info.json     # Verified split sizes and schema
│   ├── eda_stats.json        # Class counts and token percentiles
│   ├── majority_baseline.json# Majority class baseline metrics
│   ├── figures/              # Class distribution and token length plots
│   ├── preprocessing_benchmark.json # Phase 4 fit-time and peak-RAM measurement
│   ├── metrics.csv          # Validation/test aggregate metrics
│   └── smoke/               # Ignored CPU smoke-test outputs
├── docs/                     # Documentation and audit logs
│   ├── Guidelines.pdf        # University guidelines
│   ├── OPEN_ISSUES.md        # Ambiguity register
│   ├── reference_notes.md    # Reference paper critique
│   ├── REQUIREMENTS_CHECKLIST.md # Tracked requirements
│   ├── DATASET_DECISION.md   # Dataset selection & verification
│   └── COLAB_RUNBOOK.md      # Instructions for Colab GPU execution
└── tests/                    # Pytest test suite
```

## Exact model configurations used so far

The settings below are the executed CPU-feasible settings, not a claim that
they are optimal. The final logistic-loss model is an `SGDClassifier`; the
earlier saga run is retained separately as
`tfidf_lr_saga_underconverged`. Saved evidence is in
`results/converged_lr_selection.json` and `results/classical_selection.json`.

- TF-IDF: `max_features=10_000`, word/bigram `(1, 2)`, `min_df=2`,
  `max_df=0.98`, `sublinear_tf=True`, `float32`, numeric tokens dropped.
- Logistic-loss classifier: `SGDClassifier(loss="log_loss", penalty="l2")`,
  `alpha in {1e-6, 1e-5, 1e-4}`, selected by validation macro F1,
  `class_weight=None`, `max_iter=50`, `tol=1e-3`, `random_state=42`.
- Retained saga comparison: `solver="saga"`, `C in {0.1, 1.0}`,
  `class_weight=None`, `max_iter=5`, `tol=0.5`, `random_state=42`;
  this run is labeled under-converged.
- LinearSVC: `loss="squared_hinge"`, `C in {0.01, 0.1, 1.0}`,
  `random_state=42`.
- LDA: raw counts, `max_features=500`, `min_df=5`, `max_df=0.98`,
  `max_iter=2`, `learning_method="batch"`, `K in {10, 20, 30, 40}`.
  The selected `K=40` is at the edge of the sweep grid.
- Doc2Vec: PV-DM (`dm=1`), `vector_size=100`, `window=5`, `min_count=2`,
  `epochs=5`, one worker, first 500 normalized tokens only, seed 42.

Selected hyperparameters and validation scores are saved in
`results/classical_selection.json` and `results/topic_selection.json`.
Phases 9–12 are not yet done.

The Phase 8 final metrics are in `results/metrics.csv`. Phase 8 regenerated
saved prediction archives without retraining for evaluation; the Doc2Vec
metric changed slightly because repeated Doc2Vec/BLAS execution is not
guaranteed bit-exact. See `docs/RUN_LOG.md`.

### Model artifacts

Optional demo weights are hosted in the Hugging Face repository
`AhanaSrinivas/legal-text-classification-artifacts`: the Legal-BERT state file
and the fitted scikit-learn artifacts. Download them without writing the
token to disk:

```powershell
$env:HF_ARTIFACT_REPO="<hf-username>/legal-text-classification-artifacts"
$env:HF_TOKEN="<token>"
& ".\venv\Scripts\python.exe" scripts\download_artifacts.py
```

If the token, network, or artifact is unavailable, the demo continues without
the missing file. The CPU fallback is:

```powershell
& ".\venv\Scripts\python.exe" scripts\make_demo_artifacts.py
```

## Reproduce so far (PowerShell)

```powershell
& "C:\Program Files\Python313\python.exe" -m venv venv
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\venv\Scripts\python.exe" scripts\01_run_eda.py
& ".\venv\Scripts\python.exe" scripts\benchmark_preprocessing.py
& ".\venv\Scripts\python.exe" scripts\train_classical.py
& ".\venv\Scripts\python.exe" scripts\train_topic.py
& ".\venv\Scripts\python.exe" -m pytest tests
```

The Transformer outputs come from the completed Colab GPU run and are not
reproducible on this CPU-only machine.

---

## 5. Quickstart & Installation
```bash
# 1. Clone repository
git clone <your-repo-url>
cd legal-text-classification

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run EDA & verify data
python scripts/01_run_eda.py

# 5. Run Transformer smoke test (full training is planned for GPU)
python scripts/04_train_transformer.py --smoke_test
```
For running the full Legal-BERT model on Google Colab or Kaggle GPU, see [`docs/COLAB_RUNBOOK.md`](docs/COLAB_RUNBOOK.md).
