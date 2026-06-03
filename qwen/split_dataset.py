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
    n = len(item.get("aspect_list") or [])
    if n <= 3: return 0
    if n <= 6: return 1
    return 2

def split(items: list, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    buckets: dict[int, list] = defaultdict(list)
    for item in items:
        buckets[bucket(item)].append(item)

    train, val, test = [], [], []
    for b, group in sorted(buckets.items()):
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, round(len(group) * val_ratio)) if val_ratio > 0 else 0
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test+n_val])
        train.extend(group[n_test+n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(DATA_JSON, encoding="utf-8") as f:
        items = json.load(f)

    available = [
        item for item in items
        if Path(PROJECT_ROOT / item["wav_path"].lstrip("./")).exists()
    ]

    train, val, test = split(available, args.val_ratio, args.test_ratio, args.seed)

    print(f"Total available : {len(available)}")
    print(f"Train           : {len(train)}  ({len(train)/len(available)*100:.1f}%)")
    print(f"Val             : {len(val)}   ({len(val)/len(available)*100:.1f}%)")
    print(f"Test            : {len(test)}   ({len(test)/len(available)*100:.1f}%)")

    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    
    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
