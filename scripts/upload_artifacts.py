import getpass
import os
from huggingface_hub import HfApi

repo = os.getenv(
    "HF_ARTIFACT_REPO",
    "AhanaSrinivas/legal-text-classification-artifacts",
)
token = os.getenv("HF_TOKEN") or getpass.getpass("Hugging Face token: ")
api = HfApi(token=token)
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(
    folder_path="models/demo",
    repo_id=repo,
    path_in_repo="sklearn_artifacts/demo",
    allow_patterns=["*.joblib"],
)
print("done")