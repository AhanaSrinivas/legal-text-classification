# Serialized Model Directory

This directory stores model artifacts. Large checkpoints and binary models are excluded from version control via `.gitignore`.

## Model regeneration

The fitted TF-IDF, LinearSVC, Logistic Regression, LDA, and Doc2Vec models
are not committed. For the Phase 9 CPU demo, generate only the required
classical artifacts with:

```powershell
& ".\venv\Scripts\python.exe" scripts\make_demo_artifacts.py
```

The Legal-BERT checkpoint is not in this repository. It was trained on a
Colab Tesla T4; the future demo must degrade gracefully when it is absent.
