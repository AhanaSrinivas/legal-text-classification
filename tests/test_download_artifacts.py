import os
import subprocess
import sys


def test_download_artifacts_without_environment_exits_cleanly():
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("HF_ARTIFACT_REPO", None)
    result = subprocess.run(
        [sys.executable, "scripts/download_artifacts.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0


def test_download_requests_exact_huggingface_paths(monkeypatch, tmp_path):
    import src.predict as predict

    requested = []

    def fake_download(repo_id, filename, token):
        requested.append((repo_id, filename, token))
        source = tmp_path / filename.replace("/", "_")
        source.write_bytes(b"artifact")
        return str(source)

    monkeypatch.setattr(predict, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_ARTIFACT_REPO", "owner/artifacts")
    monkeypatch.chdir(tmp_path)
    for filename in predict.ARTIFACTS:
        predict.ensure_artifact(filename)

    assert [item[1] for item in requested] == [
        "sklearn_artifacts/demo/tfidf_vectorizer.joblib",
        "sklearn_artifacts/demo/tfidf_linear_svc.joblib",
        "sklearn_artifacts/demo/tfidf_logistic_regression.joblib",
        "legal_bert_state.pt",
    ]
