"""Optional model-artifact retrieval for the local demo."""

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "AhanaSrinivas/legal-text-classification-artifacts"
ARTIFACT_SPECS = {
    "sklearn_artifacts/demo/tfidf_vectorizer.joblib": "models/demo/tfidf_vectorizer.joblib",
    "sklearn_artifacts/demo/tfidf_linear_svc.joblib": "models/demo/tfidf_linear_svc.joblib",
    "sklearn_artifacts/demo/tfidf_logistic_regression.joblib": "models/demo/tfidf_logistic_regression.joblib",
    "legal_bert_state.pt": "models/legal_bert_state.pt",
}
ARTIFACTS = tuple(ARTIFACT_SPECS)


def ensure_artifact(filename: str) -> Path | None:
    """Ensure one optional artifact exists locally, without failing the demo."""
    if filename not in ARTIFACT_SPECS:
        raise ValueError(f"Unknown artifact filename: {filename}")
    hf_path = filename
    local_path = Path(ARTIFACT_SPECS[hf_path])
    if local_path.exists():
        return local_path

    token = os.getenv("HF_TOKEN")
    if not token:
        print(f"Optional artifact unavailable (HF_TOKEN is not set): {hf_path}")
        return None

    repo_id = os.getenv("HF_ARTIFACT_REPO", DEFAULT_REPO)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            token=token,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local_path)
    except Exception as exc:
        print(f"Optional artifact unavailable at HF path {hf_path}: {exc}")
        return None
    return local_path
