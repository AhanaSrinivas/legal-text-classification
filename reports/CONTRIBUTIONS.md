# Team contributions

| Member | Name / SRN |
|---|---|
| Member 1 | [Member 1 Name / SRN] |
| Member 2 | [Member 2 Name / SRN] |

## Split of work

As agreed by the team:

- **Member 1:** phases 1–8: requirements, design, dataset and EDA,
  preprocessing, classical models, LDA and Doc2Vec, Legal-BERT training,
  evaluation and error analysis.
- **Member 2:** phases 9–12: Streamlit demo, additional tests, README results,
  two-page write-up, slides, viva notes, final validation.

The Git history is the record of who committed what:

```bash
git shortlog -sne            # commits per author
git log --author="<name>" --stat
```

## Components

Fill in the owner column (Member 1, Member 2 or both) before submission.

| Component | Main files | Phase | Owner |
|---|---|---|---|
| Requirements analysis and plan | `docs/REQUIREMENTS_CHECKLIST.md`, `docs/OPEN_ISSUES.md`, `docs/reference_notes.md`, `PROJECT_PLAN.md` | 1–2 | |
| Dataset choice, loading, audits, EDA | `src/data.py`, `scripts/01_run_eda.py`, `docs/DATASET_DECISION.md`, `results/dataset_info.json`, `results/eda_stats.json` | 3 | |
| Preprocessing and leakage control | `src/preprocess.py`, `scripts/benchmark_preprocessing.py`, `docs/PREPROCESSING_DECISIONS.md`, `tests/test_preprocess.py`, `tests/test_leakage.py` | 4 | |
| TF-IDF + LinearSVC / LR models | `scripts/train_classical.py`, `scripts/train_converged_lr.py` | 5 | |
| LDA + LR, Doc2Vec + LR | `scripts/train_topic.py` | 6 | |
| Legal-BERT (Colab GPU) | `scripts/04_train_transformer.py`, `docs/COLAB_RUNBOOK.md`, `reports/colab_run_log.ipynb`, `reports/colab_run2_demo_weights.ipynb` | 7 | |
| Evaluation, CIs, error analysis | `src/evaluate.py`, `scripts/evaluate_all.py`, `results/error_analysis.md`, `docs/RUN_LOG.md` | 8 | |
| Model artifacts on Hugging Face | `src/predict.py`, `scripts/download_artifacts.py`, `scripts/upload_artifacts.py`, `scripts/make_demo_artifacts.py` | 8 | |
| Streamlit demo | `app.py`, `src/demo.py`, `scripts/check_demo_artifacts.py` | 9 | |
| Additional tests and test report | `tests/test_evaluate.py`, `tests/test_results_integrity.py`, `tests/test_data.py`, `tests/test_models.py`, `tests/test_demo.py`, `tests/test_app.py`, `results/test_report.txt` | 10 | |
| README, write-up, slides, viva notes | `README.md`, `scripts/make_readme_results.py`, `reports/writeup.py`, `scripts/make_slides.py`, `scripts/make_viva_qa.py`, `src/report_facts.py`, `docs/CITATIONS.md` | 11 | |
| Final validation and number audit | `run_all.py`, `scripts/audit_numbers.py`, `docs/FINAL_VALIDATION.md` | 12 | |

## Use of AI assistance

AI coding assistants were used during the project to draft code, tests and
documentation. Commits made with an assistant carry a `Co-Authored-By` trailer.
The team reviewed the generated code and text, ran it, and is responsible for
it; each member should be able to explain every file they own. No result was
produced by an AI tool: every number in the README, write-up, slides and viva
notes is generated from the files in `results/` and checked by
`scripts/audit_numbers.py`.
