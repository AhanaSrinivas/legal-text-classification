# University Mini-Project Requirements Checklist
**Course**: UE24CS352A - Machine Learning  
**Project Title**: Classification of Legal Text  
**Submission Deadline**: Saturday, October 10, 2026 (11:59 PM)  

| # | Requirement | Source | Planned Artifact | Status | Evidence / Verification Method |
|:---|:---|:---|:---|:---|:---|
| 1 | **Team Composition** (2 students) & Role Allocation | `Guidelines.pdf` §2 | `reports/CONTRIBUTIONS.md` | PLANNED | Template with placeholders `[Member 1 Name / SRN]` and `[Member 2 Name / SRN]`. |
| 2 | **Private GitHub Repository** setup with clean history | `Guidelines.pdf` §3.A, User Rule 5 | Git repository (`.git/`, `.gitignore`) | PARTIAL | Local `git init` and `.gitignore` configured; initial commit made; remotes/push to be managed manually by user. |
| 3 | **README.md** with setup & run commands | `Guidelines.pdf` §3.A | `README.md` | PLANNED | Setup, venv activation, single-command run instructions, architecture summary, paper comparison table. |
| 4 | **Two-Page Write-up (PDF)** covering problem, dataset, approach, implementation, conclusions | `Guidelines.pdf` §3.B | `reports/writeup.pdf` | PLANNED | Programmatically generated PDF; automated script enforcing $\le 2$ pages; editable source in `reports/writeup.py`. |
| 5 | **Five Mandatory Sections** in write-up | `Guidelines.pdf` §3.B | `reports/writeup.pdf` | PLANNED | Explicit section headers matching university syllabus requirements. |
| 6 | **Slide Deck for Review** (~15 slides with speaker notes) | `Guidelines.pdf` §4 | `reports/slides.pptx` | PLANNED | Automated generation script (`scripts/make_slides.py`) via `python-pptx` with embedded figures and speaker notes. |
| 7 | **Live Interactive Demonstration** | `Guidelines.pdf` §4 | `app.py` (Streamlit) | PLANNED | Web demo accepting legal opinions, showing category predictions, confidence breakdown, and real test examples. |
| 8 | **Viva Preparation & Defense Support** | `Guidelines.pdf` §4 | `reports/VIVA_QA.md` | PLANNED | Comprehensive Q&A covering ML theory, code walkthroughs, math, evaluation metrics, and design tradeoffs. |
| 9 | **Evaluation Criteria Alignment** (10 marks rubric) | `Guidelines.pdf` §5 | Full repository suite | PLANNED | Covers all 8 rubric facets: deliverables, functionality, repo maintenance, writeup, presentation, demo, Q&A, contribution. |
| 10 | **Empirical Honesty & Zero Fabrication** | User Rule 1 | `results/` JSON & CSV outputs | PLANNED | Every metric, table, and plot generated strictly by executed code saved under `results/`. |
| 11 | **Independent Results vs Paper** | User Rule 2, Context | `docs/reference_notes.md`, `README.md` | IMPLEMENTED | Documented paper's Table 1 inversion; our metrics evaluated strictly on our test split. |
| 12 | **Strict Reproducibility & Fixed Seed** | User Rule 4 | Pipeline scripts (`seed=42`) | PLANNED | Deterministic splits, model initializations, and data processing across all scripts. |
| 13 | **Standardized Public Dataset** (LexGLUE SCOTUS) | User Phase 3 | `src/data.py`, `results/dataset_info.json` | IMPLEMENTED | Verified 7,800 docs across Train (5,000), Val (1,400), Test (1,400); 13 SCDB classes; zero cross-split duplicate leaks. |
| 14 | **Leakage-Free Preprocessing** | User Phase 4 | `src/preprocess.py`, `docs/PREPROCESSING_DECISIONS.md` | PLANNED | Vectorizers/models fitted only on train split; legal-specific stopword handling. |
| 15 | **Classical ML Baselines** (TF-IDF + LR, TF-IDF + LinearSVC) | User Phase 5 | `src/models_classical.py`, `scripts/train_classical.py` | PLANNED | Parameter tuning on validation split; calibrated/decision scores for SVM; metrics saved to `results/`. |
| 16 | **Paper Methods Reproduction** (LDA + LR, Doc2Vec + LR) | User Phase 6 | `src/models_topic.py`, `scripts/train_topic.py` | PLANNED | LDA topic sweep on validation split; Doc2Vec inference on val/test; topic keyword extraction. |
| 17 | **Working Transformer Extension** (Legal-BERT) | User Phase 7 | `src/models_transformer.py`, `scripts/train_transformer.py` | PLANNED | Honest 512-token truncation; hardware logging (`results/transformer_run_info.json`); memory-safe batching. |
| 18 | **Unified Evaluation & Error Analysis** | User Phase 8 | `src/evaluate.py`, `results/error_analysis.md` | PLANNED | Shared test set evaluation; Macro/Weighted F1; confusion matrices; 8-10 qualitative error case studies. |
| 19 | **Automated Unit Testing Suite** | User Phase 10 | `tests/` (`pytest`) | PLANNED | Tests data loading, preprocessing, split integrity, small model fitting, metric calculation, and app startup. |
| 20 | **Single-Command Pipeline Runner** | User Phase 12 | `run_all.py` / `Makefile` | PLANNED | End-to-end execution script automating training, evaluation, report generation, and verification. |
