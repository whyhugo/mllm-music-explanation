"""
Train / Validation / Test split for MusicCaps.

Reads musiccaps_processed.json, performs a reproducible stratified-by-aspect
random split, and writes:
    data/musiccaps_train.json
    data/musiccaps_val.json
    data/musiccaps_test.json

Usage
-----
python data/split.py                          # 80/10/10 split, seed 42
python data/split.py --val_ratio 0.15 --test_ratio 0.15  # 70/15/15 split
python data/split.py --seed 0                # different shuffle
python data/split.py --stats                 # print aspect distribution
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_JSON     = PROJECT_ROOT / "musiccaps_processed.json"
TRAIN_JSON   = PROJECT_ROOT / "data" / "musiccaps_train.json"
VAL_JSON     = PROJECT_ROOT / "data" / "musiccaps_val.json"
TEST_JSON    = PROJECT_ROOT / "data" / "musiccaps_test.json"


# ── Helpers ──────────────────────────────────────────────────────────────────
def top_aspects(data: list[dict], n: int = 15) -> list[str]:
    """Return the N most common aspect tags across the dataset."""
    counter: Counter = Counter()
    for item in data:
        for tag in item.get("aspect_list", []):
            counter[tag.lower().strip()] += 1
    return [tag for tag, _ in counter.most_common(n)]


def aspect_bucket(item: dict, top: list[str]) -> str:
    """Map a sample to its primary (most-common) aspect bucket for stratification."""
    tags = {t.lower().strip() for t in item.get("aspect_list", [])}
    for t in top:            # top is ordered most→least common
        if t in tags:
            return t
    return "__other__"


def stratified_split(
    data: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data into train / val / test with approximate stratification on top aspects.
    Guarantees at least 1 sample per bucket in val and test if the bucket is large enough.
    """
    rng    = random.Random(seed)
    top    = top_aspects(data)
    buckets: dict[str, list[dict]] = {}
    for item in data:
        key = aspect_bucket(item, top)
        buckets.setdefault(key, []).append(item)

    train, val, test = [], [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) >= 3 else 0
        n_test = max(1, round(len(items) * test_ratio)) if len(items) >= 3 else 0
        val.extend(items[:n_val])
        test.extend(items[n_val:n_val + n_test])
        train.extend(items[n_val + n_test:])

    # Final shuffle so order isn't bucketed
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def print_stats(label: str, data: list[dict]) -> None:
    top = top_aspects(data, n=8)
    counts = Counter()
    for item in data:
        for tag in item.get("aspect_list", []):
            if tag.lower().strip() in top:
                counts[tag.lower().strip()] += 1

    print(f"\n  {label}  ({len(data)} samples)")
    for tag in top:
        bar = "█" * (counts[tag] * 20 // max(1, len(data)))
        print(f"    {tag:<30} {counts[tag]:>4}  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",       default=str(SRC_JSON), help="Source JSON manifest")
    parser.add_argument("--train_out", default=str(TRAIN_JSON))
    parser.add_argument("--val_out",   default=str(VAL_JSON))
    parser.add_argument("--test_out",  default=str(TEST_JSON))
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Fraction for validation (default: 0.10)")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Fraction for testing (default: 0.10)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--stats",     action="store_true", help="Print aspect distribution")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.src}")

    # Filter to items whose audio actually exists
    available = [
        item for item in data
        if Path(PROJECT_ROOT / item["wav_path"].lstrip("./")).exists()
    ]
    missing = len(data) - len(available)
    if missing:
        print(f"  ⚠  {missing} items skipped (audio not found on disk)")

    train, val, test = stratified_split(available, args.val_ratio, args.test_ratio, args.seed)

    # Save
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    for path, split in [(args.train_out, train), (args.val_out, val), (args.test_out, test)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=4, ensure_ascii=False)

    print(f"\nSplit summary  (seed={args.seed}, val_ratio={args.val_ratio}, test_ratio={args.test_ratio})")
    print(f"  Train : {len(train):>4} samples  →  {args.train_out}")
    print(f"  Val   : {len(val):>4} samples  →  {args.val_out}")
    print(f"  Test  : {len(test):>4} samples  →  {args.test_out}")
    print(f"  Total : {len(train)+len(val)+len(test):>4}  (original: {len(data)})")

    if args.stats:
        print_stats("TRAIN", train)
        print_stats("VAL",   val)
        print_stats("TEST",  test)


if __name__ == "__main__":
    main()
