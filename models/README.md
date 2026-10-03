# Serialized Model Directory

This directory stores model artifacts. Large checkpoints and binary models are excluded from version control via `.gitignore`.

## Model Regeneration Commands
- **TF-IDF + Classical Models (M1 & M2)**:
  ```bash
  python scripts/02_train_classical.py
  ```
- **LDA Topic & Doc2Vec Models (M3 & M4)**:
  ```bash
  python scripts/03_train_topic.py
  ```
- **Legal-BERT Fine-Tuned Model (M5)**:
  ```bash
  python scripts/04_train_transformer.py
  ```
  *(Or run on Google Colab/Kaggle GPU following `docs/COLAB_RUNBOOK.md`)*
