"""Optional model-artifact retrieval for the local demo."""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO = "AhanaSrinivas/legal-text-classification-artifacts"
ARTIFACTS = (
    "legal_bert_state.pt",
    "sklearn_artifacts/tfidf_vectorizer.joblib",
    "sklearn_artifacts/tfidf_linear_svc.joblib",
    "sklearn_artifacts/tfidf_logistic_regression.joblib",
)


def ensure_artifact(filename: str) -> Path | None:
    """Ensure one optional artifact exists locally, without failing the demo."""
    local_path = Path("models") / filename
    if local_path.exists():
        return local_path

    token = os.getenv("HF_TOKEN")
    if not token:
        LOGGER.warning(
            "Optional artifact %s is unavailable: HF_TOKEN is not set; "
            "continuing without downloaded artifacts.",
            filename,
        )
        return None

    repo_id = os.getenv("HF_ARTIFACT_REPO", DEFAULT_REPO)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir="models",
        )
    except Exception as exc:
        LOGGER.warning(
            "Optional artifact %s is unavailable from %s (%s); "
            "continuing with local fallback.",
            filename,
            repo_id,
            exc,
        )
        return None
    return Path(downloaded)
