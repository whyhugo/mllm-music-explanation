"""
Train / Val / Test split for musiccaps_processed.json
======================================================
Splits the manifest into stratified train, val, and test sets, then writes:
  data/train.json
  data/val.json
  data/test.json

Stratification is by aspect_list length bucket so that easy (few aspects)
and complex (many aspects) captions are represented in all splits.

Usage
-----
# Default 80/10/10 split, reproducible seed
python qwen/split_dataset.py

# Custom ratio and seed
python qwen/split_dataset.py --val_ratio 0.15 --test_ratio 0.15 --seed 123

# Inspect only (no files written)
python qwen/split_dataset.py --dry_run
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON    = PROJECT_ROOT / "musiccaps_processed.json"
DATA_DIR     = PROJECT_ROOT / "data"


def bucket(item: dict) -> int:
    """Return aspect_list-length bucket (0-2) for stratification."""
    n = len(item.get("aspect_list") or [])
    if n <= 3:
        return 0
    if n <= 6:
        return 1
    return 2


def split(items: list, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)

    # Group by bucket
    buckets: dict[int, list] = defaultdict(list)
    for item in items:
        buckets[bucket(item)].append(item)

    train, val, test = [], [], []
    for b, group in sorted(buckets.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        n_test = max(1, round(len(group) * test_ratio))
        val.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])
        train.extend(group[n_val + n_test:])

    # Final shuffle to avoid bucket ordering artefacts
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.10,
                        help="Fraction of data for validation (default: 0.10)")
    parser.add_argument("--test_ratio", type=float, default=0.10,
                        help="Fraction of data for testing (default: 0.10)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--input",     default=str(DATA_JSON),
                        help="Source JSON manifest")
    parser.add_argument("--out_dir",   default=str(DATA_DIR),
                        help="Output directory for train.json, val.json, and test.json")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)

    # Filter to items whose audio actually exists
    available = [
        item for item in items
        if Path(PROJECT_ROOT / item["wav_path"].lstrip("./")).exists()
    ]
    missing = len(items) - len(available)
    if missing:
        print(f"  ⚠  {missing} items skipped (audio not found on disk)")

    train, val, test = split(available, args.val_ratio, args.test_ratio, args.seed)

    print(f"Total available : {len(available)}")
    train_pct = (len(train) / len(available) * 100) if available else 0.0
    val_pct   = (len(val) / len(available) * 100) if available else 0.0
    test_pct  = (len(test) / len(available) * 100) if available else 0.0
    print(f"Train           : {len(train)}  ({train_pct:.1f}%)")
    print(f"Val             : {len(val)}   ({val_pct:.1f}%)")
    print(f"Test            : {len(test)}   ({test_pct:.1f}%)")
    print(f"Val ratio       : {args.val_ratio}  test ratio : {args.test_ratio}  seed={args.seed}")

    # Bucket distribution check
    for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
        counts = defaultdict(int)
        for item in split_items:
            counts[bucket(item)] += 1
        dist = "  ".join(f"b{b}={counts[b]}" for b in sorted(counts))
        print(f"  {split_name} distribution: {dist}")

    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.json"
    val_path   = out / "val.json"
    test_path  = out / "test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=4, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {train_path}  ({len(train)} items)")
    print(f"  {val_path}  ({len(val)} items)")
    print(f"  {test_path}  ({len(test)} items)")
    print("\nTo use in training, set DATA_JSON in train_lora_qwen.py:")
    print(f'  DATA_JSON = os.path.join(PROJECT_ROOT, "data", "train.json")')


if __name__ == "__main__":
    main()
