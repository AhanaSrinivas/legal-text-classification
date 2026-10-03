# Classification of Legal Text (US Supreme Court Opinions)
**Course**: UE24CS352A - Machine Learning Mini-Project  
**Domain**: Legal Natural Language Processing (NLP)  
**Dataset**: LexGLUE SCOTUS Benchmark (`coastalcph/lex_glue`, `scotus` config)  

---

## 1. Project Overview & Motivation
Classifying legal texts is a core challenge in legal informatics. Legal opinions are characterized by long-running syntactic constructions, extensive citations to judicial precedent, and non-standard domain jargon. Automated categorization of court decisions into human-curated taxonomy areas (such as the Supreme Court Database issue areas) enables streamlined case-law research, assists public defenders, and democratizes access to justice.

This project reproduces and extends the methodology investigated by Krithika Iyer (Stanford CS229, 2020, *Classification of Legal Text*):
- **Paper Methods**: Evaluates Latent Dirichlet Allocation (LDA) topic mixtures + Logistic Regression, and Paragraph Vector (Doc2Vec) dense embeddings + Logistic Regression.
- **Our Baselines & Extensions**: Implements strong TF-IDF classical ML baselines (Logistic Regression and LinearSVC), resolves data leakage via strict split isolation, fine-tunes domain-adapted Transformers (`nlpaueb/legal-bert-base-uncased`), provides honest empirical reporting on long-document truncation (512 tokens), and provides an interactive Streamlit application.

---

## 2. Paper Method vs. Our Project Strategy

| Component | Reference Paper (Iyer 2020) | Our Implementation & Reproduction | Status / Notes |
|:---|:---|:---|:---|
| **Dataset Source** | `textacy` scrape (~8,200 opinions) | LexGLUE SCOTUS (`coastalcph/lex_glue`) | Verified public benchmark, non-leaking splits |
| **Splitting Method** | Random unseeded split | Chronological split recorded by LexGLUE metadata (1946–1982 train, 1982–1991 val, 1991–2016 test) | Intended to evaluate temporal generalization; the exact boundary dates are attributed to the LexGLUE dataset card, not the paper |
| **Class Taxonomy** | 15 issue areas / 279 issue codes | 13 active SCDB issue areas (0-12) | Class 14 ('Private Action') is absent from the adopted LexGLUE SCOTUS corpus; see `docs/DATASET_DECISION.md` |
| **Classical Baselines**| None | TF-IDF + Logistic Regression, TF-IDF + LinearSVC | Essential classical benchmarks |
| **Topic Modeling** | LDA (TF-IDF input) + Logistic Regression | LDA (Count input, topic sweep $K \in \{10,20,30,40\}$) + LR | Primary input uses count frequencies; TF-IDF difference noted |
| **Document Vectors** | Doc2Vec + Logistic Regression | Doc2Vec (PV-DBOW / PV-DM, fit strictly on train) + LR | Exact inference hygiene on val/test |
| **Transformer** | Failed (Ran out of memory on Colab; no results) | Fine-tuned `nlpaueb/legal-bert-base-uncased` (planned full GPU run) | Only a 20-sample CPU smoke test is currently recorded; it is not an experimental result |
| **Interactive Demo**| None | Streamlit Web Application (`app.py`) | Interactive opinion classifier with probability breakdown |
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
│   ├── models_classical.py   # (planned) Baseline Logistic Regression and LinearSVC
│   ├── models_topic.py       # (planned) LDA and Doc2Vec models
│   ├── models_transformer.py # (planned) Transformer fine-tuning module
│   ├── evaluate.py           # (planned) Unified evaluation routines
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
│   ├── transformer_predictions.npz # Smoke-test placeholder; not reportable until replaced by a GPU run
│   └── transformer_run_info.json   # Smoke-test provenance; not reportable until replaced by a GPU run
├── docs/                     # Documentation and audit logs
│   ├── Guidelines.pdf        # University guidelines
│   ├── OPEN_ISSUES.md        # Ambiguity register
│   ├── reference_notes.md    # Reference paper critique
│   ├── REQUIREMENTS_CHECKLIST.md # Tracked requirements
│   ├── DATASET_DECISION.md   # Dataset selection & verification
│   └── COLAB_RUNBOOK.md      # Instructions for Colab GPU execution
└── tests/                    # Pytest test suite
```

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
