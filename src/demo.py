"""Helpers for the Streamlit demo in app.py.

Everything here is importable without Streamlit so it can be unit-tested.
Live predictions use the optional artifacts under models/; stored results
are read from results/ and are never recomputed or edited here.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import SCDB_LABEL_NAMES


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SAMPLE_CASES_PATH = ROOT / "data" / "sample_cases.json"
BERT_BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
BERT_MAX_TOKENS = 512

MODEL_LABELS = {
    "tfidf_linear_svc": "TF-IDF + LinearSVC",
    "tfidf_logistic_regression": "TF-IDF + Logistic Regression (SGD, log loss)",
    "transformer_legal_bert": "Legal-BERT (first 512 tokens)",
    "lda_lr": "LDA topics + Logistic Regression",
    "doc2vec_lr": "Doc2Vec + Logistic Regression",
    "tfidf_lr_saga_underconverged": "TF-IDF + LR (saga, under-converged)",
    "majority_baseline": "Majority-class baseline",
}
# Models that can run live in the demo, in display order.
LIVE_MODELS = ("tfidf_linear_svc", "tfidf_logistic_regression", "transformer_legal_bert")


def models_dir() -> Path:
    """Artifact directory; LTC_MODELS_DIR overrides it (used by the tests)."""
    return Path(os.getenv("LTC_MODELS_DIR", ROOT / "models"))


def artifact_paths() -> dict:
    base = models_dir()
    return {
        "tfidf_vectorizer": base / "demo" / "tfidf_vectorizer.joblib",
        "tfidf_linear_svc": base / "demo" / "tfidf_linear_svc.joblib",
        "tfidf_logistic_regression": base / "demo" / "tfidf_logistic_regression.joblib",
        "transformer_legal_bert": base / "legal_bert_state.pt",
    }


def available_models() -> list:
    """Live models whose artifacts exist locally."""
    paths = artifact_paths()
    found = []
    for name in LIVE_MODELS:
        if name == "transformer_legal_bert":
            ok = paths[name].exists()
        else:
            ok = paths[name].exists() and paths["tfidf_vectorizer"].exists()
        if ok:
            found.append(name)
    return found


# ---------------------------------------------------------------- classical


def load_sklearn_model(name: str):
    """Return (vectorizer, classifier) for a TF-IDF model, or None if absent."""
    import joblib

    paths = artifact_paths()
    if not (paths[name].exists() and paths["tfidf_vectorizer"].exists()):
        return None
    return joblib.load(paths["tfidf_vectorizer"]), joblib.load(paths[name])


def predict_sklearn(vectorizer, classifier, text: str) -> dict:
    """Predict one document; probabilities if the model has them, else decision scores."""
    features = vectorizer.transform([text])
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(features)[0]
        score_kind = "probability"
    else:
        scores = classifier.decision_function(features)[0]
        score_kind = "decision score"
    scores = _full_class_scores(classifier.classes_, scores)
    return {
        "label": int(np.argmax(scores)),
        "scores": scores,
        "score_kind": score_kind,
        "vocabulary_hits": int(features.nnz),
    }


def reported_setting_note(name: str, classifier, results_dir=RESULTS_DIR):
    """Caption if the demo artifact's key hyperparameter differs from the reported model."""
    results_dir = Path(results_dir)
    if name == "tfidf_linear_svc":
        reported = json.loads((results_dir / "classical_selection.json").read_text())
        reported_c = reported["tfidf_linear_svc"]["params"][0]
        if classifier.C != reported_c:
            return (f"Demo artifact uses C={classifier.C}; the reported LinearSVC used "
                    f"C={reported_c} (selected on validation), so live scores can differ.")
    if name == "tfidf_logistic_regression":
        reported = json.loads((results_dir / "converged_lr_selection.json").read_text())
        if classifier.alpha != reported["selected_alpha"]:
            return (f"Demo artifact uses alpha={classifier.alpha}; the reported model used "
                    f"alpha={reported['selected_alpha']}.")
    return None


def _full_class_scores(classes, scores) -> np.ndarray:
    """Place per-class scores in label order 0..12 (missing classes get -inf)."""
    full = np.full(len(SCDB_LABEL_NAMES), -np.inf)
    full[np.asarray(classes, dtype=int)] = scores
    return full


# ---------------------------------------------------------------- Legal-BERT


def load_bert(state_path=None):
    """Build Legal-BERT from its config and load the fine-tuned demo weights.

    Only the tokenizer and config are fetched from the Hugging Face hub (and
    cached); the 13-class weights come from models/legal_bert_state.pt.
    Returns (tokenizer, model) or None if the state file is absent.
    """
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    state_path = Path(state_path or artifact_paths()["transformer_legal_bert"])
    if not state_path.exists():
        return None
    config = AutoConfig.from_pretrained(
        BERT_BASE_MODEL,
        num_labels=len(SCDB_LABEL_NAMES),
        id2label=dict(enumerate(SCDB_LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(SCDB_LABEL_NAMES)},
    )
    model = AutoModelForSequenceClassification.from_config(config)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)
    return tokenizer, model


def count_bert_tokens(tokenizer, text: str) -> int:
    """Number of word-piece tokens in the whole text (no truncation)."""
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def predict_bert(tokenizer, model, text: str) -> dict:
    """Softmax probabilities from the first 512 tokens, as in training."""
    import torch

    encoded = tokenizer(text, truncation=True, max_length=BERT_MAX_TOKENS, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    probabilities = torch.softmax(logits, dim=-1).numpy()
    return {
        "label": int(np.argmax(probabilities)),
        "scores": probabilities,
        "score_kind": "probability",
    }


def truncation_note(total_tokens: int, limit: int = BERT_MAX_TOKENS):
    """Warning text when Legal-BERT cannot see the whole input, else None."""
    usable = limit - 2  # [CLS] and [SEP]
    if total_tokens <= usable:
        return None
    share = 100 * usable / total_tokens
    return (
        f"Legal-BERT reads only the first {limit} tokens. This text has "
        f"{total_tokens:,} tokens, so it sees about {share:.0f}% of it."
    )


# ---------------------------------------------------------------- shared


def top_k(scores, k: int = 5) -> pd.DataFrame:
    """Top-k classes by score, highest first."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)[:k]
    return pd.DataFrame({
        "Issue area": [SCDB_LABEL_NAMES[i] for i in order],
        "Score": [float(scores[i]) for i in order],
    })


def load_sample_cases(path=SAMPLE_CASES_PATH) -> list:
    """The committed fixture: one seeded test case per class, first 3,000 characters."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stored_predictions(test_index: int, results_dir=RESULTS_DIR) -> pd.DataFrame:
    """Saved full-document test predictions for one test case, per model."""
    rows = []
    for name in ("tfidf_linear_svc", "tfidf_logistic_regression", "transformer_legal_bert",
                 "lda_lr", "doc2vec_lr"):
        path = Path(results_dir) / "predictions" / f"{name}.npz"
        if not path.exists():
            continue
        archive = np.load(path, allow_pickle=True)
        true_label = int(archive["test_labels"][test_index])
        predicted = int(archive["test_predictions"][test_index])
        rows.append({
            "Model": MODEL_LABELS[name],
            "Stored prediction": SCDB_LABEL_NAMES[predicted],
            "Correct": predicted == true_label,
        })
    return pd.DataFrame(rows)


def metrics_table(split: str = "test", results_dir=RESULTS_DIR) -> pd.DataFrame:
    """Aggregate metrics from results/metrics.csv with bootstrap CIs (test only)."""
    results_dir = Path(results_dir)
    metrics = pd.read_csv(results_dir / "metrics.csv")
    metrics = metrics[metrics["split"] == split].copy()
    table = pd.DataFrame({
        "Model": metrics["model"].map(MODEL_LABELS).fillna(metrics["model"]),
        "Macro F1": metrics["macro_f1"],
        "Accuracy": metrics["accuracy"],
        "Weighted F1": metrics["weighted_f1"],
    })
    if split == "test":
        ci = json.loads((results_dir / "bootstrap_ci.json").read_text())
        table["Macro F1 95% CI"] = [
            f"[{ci[m]['lower_95']:.3f}, {ci[m]['upper_95']:.3f}]" if m in ci else ""
            for m in metrics["model"]
        ]
    return table.sort_values("Macro F1", ascending=False).reset_index(drop=True)


def per_class_table(model: str, results_dir=RESULTS_DIR) -> pd.DataFrame:
    """Per-class test precision/recall/F1 with support, from results/per_class_f1.csv."""
    table = pd.read_csv(Path(results_dir) / "per_class_f1.csv")
    table = table[table["model"] == model].drop(columns="model")
    return table.rename(columns={
        "class": "Issue area", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "support": "Test support",
    }).reset_index(drop=True)
