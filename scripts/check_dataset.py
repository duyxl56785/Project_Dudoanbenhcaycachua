"""Validate immutable NV1 PlantVillage splits without changing them.

Run: python -m scripts.check_dataset [--data-root data/raw]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"Tomato___healthy": 0, "Tomato___Early_blight": 1,
          "Tomato___Late_blight": 2, "Tomato___Bacterial_spot": 3}
EXPECTED = {"train": 4115, "val": 1031, "test": 1356}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check(manifest: Path = ROOT / "data/dataset_manifest.csv",
          split_dir: Path = ROOT / "data/splits",
          data_root: Path | None = None) -> dict:
    rows = load_csv(manifest)
    if len(rows) != 6502:
        raise AssertionError(f"Expected 6502 manifest rows, got {len(rows)}")
    required = {"split", "path", "label", "group_id", "sha256"}
    if not all(required <= set(row) for row in rows):
        raise ValueError(f"Missing manifest columns from {required}")
    count, by_class = Counter(), Counter()
    path_to_split, hash_to_split, group_to_split = {}, {}, {}
    for row in rows:
        split, path, label = row["split"], row["path"], row["label"]
        if split not in EXPECTED or label not in LABELS:
            raise ValueError(f"Invalid split/class {split!r}, {label!r}")
        if LABELS.get(Path(path).parent.name) != LABELS[label]:
            raise AssertionError(f"Label does not match folder: {path}")
        for index, value in ((path_to_split, path), (hash_to_split, row["sha256"]),
                             (group_to_split, row["group_id"])):
            if not value:
                raise AssertionError(f"Empty path/hash/group for {path}")
            if value in index and index[value] != split:
                raise AssertionError(f"Cross-split overlap: {value}")
            index[value] = split
        count[split] += 1
        by_class[(split, LABELS[label])] += 1
        if data_root is not None and not (data_root / path).is_file():
            raise FileNotFoundError(f"Image missing: {data_root / path}")
    if dict(count) != EXPECTED:
        raise AssertionError(f"Unexpected split counts: {dict(count)}")
    if len(path_to_split) != len(rows) or len(hash_to_split) != len(rows):
        raise AssertionError("Duplicate path/hash in selected dataset")
    for split in EXPECTED:
        exported = load_csv(split_dir / f"{split}.csv")
        expected_rows = {(r["path"], str(LABELS[r["label"]]), r["group_id"])
                         for r in rows if r["split"] == split}
        actual_rows = {(r["path"], r["label"], r.get("group", ""))
                       for r in exported}
        if len(exported) != EXPECTED[split] or actual_rows != expected_rows:
            raise AssertionError(f"Split CSV differs from source manifest: {split}")
    return {"split_counts": dict(count), "class_counts": {
        str(label): {split: by_class[(split, label)] for split in EXPECTED}
        for label in range(4)}, "unique_paths": len(path_to_split),
        "unique_sha256": len(hash_to_split), "distinct_group_ids": len(group_to_split)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Optional extracted ZIP root; checks all 6502 paths exist")
    args = parser.parse_args()
    result = check(data_root=args.data_root)
    for split, n in result["split_counts"].items():
        print(f"{split}: {n}")
    print("Manifest/CSV/SHA/group integrity: PASS")
