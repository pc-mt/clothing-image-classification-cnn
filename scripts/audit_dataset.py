"""Audit image counts and exact duplicates across dataset splits."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--remove-cross-split-duplicates",
        action="store_true",
        help="Keep the earliest split (train, val, test) and remove later exact copies.",
    )
    args = parser.parse_args()

    counts = Counter()
    hashes = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                counts[(split, path.parent.name)] += 1
                hashes[digest(path)].append((split, path))

    for (split, class_name), count in sorted(counts.items()):
        print(f"{split:5} {class_name:14} {count:4}")

    duplicate_groups = [items for items in hashes.values() if len({x[0] for x in items}) > 1]
    print(f"Cross-split duplicate groups: {len(duplicate_groups)}")
    for items in duplicate_groups:
        ordered = sorted(items, key=lambda item: SPLIT_PRIORITY[item[0]])
        print("  " + " == ".join(str(path) for _, path in ordered))
        if args.remove_cross_split_duplicates:
            for _, duplicate in ordered[1:]:
                duplicate.unlink()
                print(f"  removed: {duplicate}")


if __name__ == "__main__":
    main()
