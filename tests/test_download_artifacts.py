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
