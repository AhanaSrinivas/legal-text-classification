"""Best-effort download of optional demo artifacts."""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import ARTIFACTS, ensure_artifact


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for filename in ARTIFACTS:
        ensure_artifact(filename)


if __name__ == "__main__":
    main()
