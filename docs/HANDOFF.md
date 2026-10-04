# Teammate handoff

## Current status

Phases 1-8 are complete. Remaining work:

- Phase 9: Streamlit application.
- Phase 10: additional tests and final test report coverage.
- Phase 11: README results table, two-page PDF, slides, VIVA_QA, and
  CONTRIBUTIONS placeholders.
- Phase 12: final validation and number-trace audit.

## Setup and reproduction

```powershell
git clone <repository-url>
cd legal-text-classification
& "C:\Program Files\Python313\python.exe" -m venv venv
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\venv\Scripts\python.exe" -m pytest tests -q
& ".\venv\Scripts\python.exe" scripts\make_demo_artifacts.py
```

The demo-artifact script creates CPU model files under ignored `models/demo/`.
The fitted TF-IDF/SVC/LR/LDA/Doc2Vec models are not in Git. The Legal-BERT
checkpoint is also not in this repository: it was trained on a Colab Tesla T4.
The Streamlit app must degrade gracefully when those files are absent, and may
show stored Legal-BERT predictions for sample test cases from
`results/predictions/` rather than pretending to run the checkpoint.

## Working rules

- Every reported number must come from `results/`; never edit metrics by hand.
- Use the test split for final reporting only.
- Do not fabricate claims, runs, citations, or dataset statistics.
- Commit using your own Git identity.
- Run `git pull --rebase` before pushing.
- Do not both edit the same file; coordinate ownership first.

## Limitations to carry into the report

- The final SGD logistic-loss classifier converged under the recorded
  `n_iter_ < max_iter` check; the older saga comparison is explicitly
  under-converged and retained separately.
- LDA and Doc2Vec use reduced CPU-feasible configurations.
- Legal-BERT truncates documents to 512 tokens and uses no class weights.
- Results use a single seed.
- The chronological split has a documented year gap/metadata limitation.
- Rare classes have tiny test support, so macro F1 is noisy.
- Bootstrap confidence intervals are unpaired and describe each model
  independently.
- Doc2Vec is not guaranteed bit-exact across runs.

See [`results/error_analysis.md`](../results/error_analysis.md) for the
real-prediction error analysis and hypotheses.
