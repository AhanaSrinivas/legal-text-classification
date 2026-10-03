# Google Colab / Kaggle GPU Execution Runbook
**Project**: Legal Text Classification (UE24CS352A)  
**Target Model**: Legal-BERT (`nlpaueb/legal-bert-base-uncased`) on LexGLUE SCOTUS  

---

## 1. Overview
Supreme Court decisions are long, and CPU execution is slow. The completed
GPU run took **1174.65 seconds (19.58 minutes) on a Tesla T4**. Its metadata
is recorded in `results/transformer_run_info.json`. The run used the script
version at commit `0b78314`; later local changes only isolate smoke outputs
and guard full-result overwrites.

This runbook explains how to run `scripts/04_train_transformer.py` on Google Colab / Kaggle, download the predictions and run metadata, and place them directly in your local `results/` folder.

---

## 2. Quick Colab Execution Steps

### Step 1: Open a New Colab Notebook with GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Runtime** -> **Change runtime type** -> Select **T4 GPU** (or any available GPU accelerator).
3. Verify GPU availability in a code cell:
   ```python
   !nvidia-smi
   ```

### Step 2: Clone or Upload the Repository
If your GitHub repository is public or you have a personal access token:
```bash
!git clone https://github.com/<your-username>/legal-text-classification.git
%cd legal-text-classification
```
*Or, upload a ZIP of the `legal-text-classification` folder using the Colab file explorer on the left, unzip it, and cd into the folder:*
```bash
!unzip legal-text-classification.zip
%cd legal-text-classification
```

### Step 3: Install Required Dependencies
```bash
!pip install -q transformers datasets scikit-learn psutil
```

### Step 4: Run Training Script
Run the full GPU fine-tuning (3 epochs, batch size 8 with gradient accumulation 2 -> effective batch size 16, fp16 enabled automatically on CUDA):
```bash
!python scripts/04_train_transformer.py --epochs 3 --batch_size 8 --gradient_accumulation_steps 2
```

*(Optional: To verify that the pipeline runs cleanly before starting the full 3 epochs, test with the 20-sample smoke test:)*
```bash
!python scripts/04_train_transformer.py --smoke_test --epochs 1
```

### Step 5: Download the Generated Results
The script produces two critical lightweight output files:
1. `results/transformer_predictions.npz` (contains test/val labels, predictions, and softmax probability distributions).
2. `results/transformer_run_info.json` (contains the exact GPU specs, runtime, and validation metrics).

In Colab, download them using Python:
```python
from google.colab import files
files.download("results/transformer_predictions.npz")
files.download("results/transformer_run_info.json")
```

### Step 6: Place Files in Local Repository
Copy the two downloaded files into your local project directory:
```
legal-text-classification/
└── results/
    ├── transformer_predictions.npz
    └── transformer_run_info.json
```
Once placed in `results/`, the downstream evaluation script (`scripts/05_evaluate_all.py`), error analysis, and the Streamlit demo (`app.py`) will automatically load these predictions without requiring local GPU training.

---

## 3. Resuming If Interrupted
The script saves checkpoints to `models/transformer/best_model.pt`. If Colab disconnects, simply add `--resume`:
```bash
!python scripts/04_train_transformer.py --epochs 3 --batch_size 8 --resume
```
