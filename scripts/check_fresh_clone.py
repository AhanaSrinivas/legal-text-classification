"""Record an isolated install/test/evaluation/report rebuild without training.

Run with .venv/bin/python scripts/check_fresh_clone.py. Keeps the temporary
clone for inspection and writes command outputs to results/validation/.
The clone shares the machine's pip and Hugging Face caches, not its venv or
model artifacts. No command contacts a Git remote or uploads anything.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.tags import parse_tag, sys_tags
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/validation/fresh_clone.json"
REQUIRED = ("metrics.csv", "per_class_f1.csv", "bootstrap_ci.json", "error_analysis.md")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_wheels(cache, target):
    """Expose intact downloaded wheels from pip's HTTP cache for --find-links."""
    target.mkdir()
    supported = set(sys_tags())
    recovered = []
    for path in sorted(cache.rglob("*.body")):
        if not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist()
                        if name.endswith(".dist-info/WHEEL") and name.count("/") == 1]
            if len(metadata) != 1:
                continue
            wheel = metadata[0]
            tags = [line.removeprefix("Tag: ") for line in archive.read(wheel).decode().splitlines()
                    if line.startswith("Tag: ")]
            compatible = [tag for tag in tags if supported.intersection(parse_tag(tag))]
            if not compatible:
                continue
            name = wheel.split(".dist-info/")[0] + "-" + compatible[0] + ".whl"
            destination = target / name
            shutil.copyfile(path, destination)
            recovered.append({"wheel": name, "sha256": digest(destination)})
    return recovered


def compare_reproduction(clone, paths):
    """Distinguish byte drift from changed image pixels or prediction arrays."""
    comparisons = {}
    for relative in paths:
        original, rebuilt = ROOT / relative, clone / relative
        item = {"original_sha256": digest(original), "rebuilt_sha256": digest(rebuilt)}
        if original.suffix == ".png":
            with Image.open(original) as a, Image.open(rebuilt) as b:
                item["pixels_equal"] = a.size == b.size and a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes()
        elif original.suffix in {".npz", ".pptx"}:
            with zipfile.ZipFile(original) as a, zipfile.ZipFile(rebuilt) as b:
                names = sorted(set(a.namelist()) | set(b.namelist()))
                item["changed_zip_members"] = [name for name in names if name not in a.namelist()
                                                or name not in b.namelist() or a.read(name) != b.read(name)]
                item["member_content_equal"] = not item["changed_zip_members"]
        comparisons[relative] = item
    return comparisons


def main():
    scratch = Path(tempfile.mkdtemp(prefix="ltc-fresh-clone-"))
    clone = scratch / "ltc"
    record = {"source": str(ROOT), "clone": str(clone), "commands": [],
              "cache_note": "Shared machine pip/Hugging Face caches; isolated venv; no demo weights copied."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def run(args, cwd=clone, *, env=None, allow_failure=False):
        label = shlex.join(map(str, args))
        print(f"RUN {label}", flush=True)
        output = scratch / "command-output.txt"
        with output.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        entry = {"command": label, "cwd": str(cwd), "exit_code": completed.returncode,
                 "output": output.read_text(encoding="utf-8", errors="replace")}
        if env is not None:
            entry["environment_overrides"] = {key: env[key] for key in
                                              ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE") if key in env}
        record["commands"].append(entry)
        save()
        print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} {label}", flush=True)
        if completed.returncode and not allow_failure:
            raise RuntimeError(f"Command failed; see {OUTPUT} and {output}")
        return entry["output"]

    try:
        record["source_commit"] = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        run(["git", "clone", str(ROOT), str(clone)], cwd=scratch)
        run([sys.executable, "-m", "venv", ".venv"])
        run([".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"], allow_failure=True)
        if record["commands"][-1]["exit_code"]:
            cache = Path(run([".venv/bin/python", "-m", "pip", "cache", "dir"]).strip())
            wheelhouse = scratch / "wheels"
            record["offline_wheels"] = cached_wheels(cache, wheelhouse)
            save()
            run([".venv/bin/python", "-m", "pip", "install", "--no-index", "--find-links",
                 str(wheelhouse), "-r", "requirements.txt"])
        run([".venv/bin/python", "-m", "pip", "check"])
        offline = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        offline.pop("LTC_MODELS_DIR", None)
        run([".venv/bin/python", "-m", "pytest", "tests", "-q", "-rs"], env=offline)
        record["before"] = {name: digest(clone / "results" / name) for name in REQUIRED}
        run([".venv/bin/python", "scripts/evaluate_all.py"], env=offline)
        record["after"] = {name: digest(clone / "results" / name) for name in REQUIRED}
        record["required_results_unchanged"] = record["before"] == record["after"]
        run(["git", "diff", "--stat"])
        record["evaluation_changes"] = run(["git", "diff", "--name-only"]).splitlines()
        run([".venv/bin/python", "run_all.py"])
        record["final_status"] = run(["git", "status", "--short"])
        changed = run(["git", "diff", "--name-only"]).splitlines()
        record["byte_comparisons"] = compare_reproduction(clone, changed)
        record["prediction_content_unchanged"] = all(
            item.get("member_content_equal", False) for name, item in record["byte_comparisons"].items()
            if name.endswith(".npz")
        )
        unexpected = [name for name in changed if not name.endswith((".png", ".npz"))
                      and name != "reports/slides.pptx"]
        record["unexpected_changes"] = unexpected
        record["passed"] = (record["required_results_unchanged"]
                            and record["prediction_content_unchanged"] and not unexpected)
        if not record["passed"]:
            raise RuntimeError("Required evaluation files changed")
    except (OSError, RuntimeError) as exc:
        record["passed"] = False
        record["error"] = str(exc)
    save()
    print(f"{'PASS' if record['passed'] else 'FAIL'} fresh clone: {OUTPUT}", flush=True)
    return int(not record["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
