"""Rebuild documents from saved results, audit numbers, and optionally run tests.

Run with .venv/bin/python run_all.py [--tests]. No training or evaluation runs.
Each builder validates its own output (including PDF page count and slide notes).
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = (
    ("README blocks", ("scripts/make_readme_results.py",)),
    ("PDF (page check)", ("reports/writeup.py",)),
    ("slides (count and notes)", ("scripts/make_slides.py",)),
    ("VIVA_QA", ("scripts/make_viva_qa.py",)),
    ("number audit", ("scripts/audit_numbers.py",)),
)


def run_step(name, arguments):
    try:
        completed = subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, text=True,
            encoding="utf-8", errors="replace", capture_output=True, check=False,
        )
    except OSError as exc:
        print(f"FAIL {name}: {exc}", flush=True)
        return False
    passed = completed.returncode == 0
    reason = "completed" if passed else f"exit status {completed.returncode}"
    print(f"{'PASS' if passed else 'FAIL'} {name}: {reason}", flush=True)
    for output in (completed.stdout, completed.stderr):
        if output.strip():
            print(output.rstrip(), flush=True)
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="also run the read-only report checks and test suite")
    args = parser.parse_args(argv)
    steps = list(STEPS)
    if args.tests:
        steps.append(("pytest", ("-m", "pytest", "tests", "-q")))
    failed = False
    for name, arguments in steps:
        if not run_step(name, arguments):
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
