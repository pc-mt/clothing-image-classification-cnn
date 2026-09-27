"""Extract the local dataset archive and remove exact cross-split duplicates."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to clothing_clean_5classes_split_audit_copy.zip")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as source:
        source.extractall(args.destination)
    dataset_dir = args.destination / "clothing_clean_5classes_split_audit_copy"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("audit_dataset.py")),
            str(dataset_dir),
            "--remove-cross-split-duplicates",
        ],
        check=True,
    )
    print(f"Dataset ready: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
