"""
Audio data augmentation for music captioning training
=====================================================
Applies pitch shifting to each training clip to create augmented variants.
Captions are reused unchanged — general music descriptions (mood, instrument,
genre) are pitch-agnostic.

Augmentation strategy
---------------------
Default: pitch shift by -2 and +2 semitones → 3× training data
  original (1054) + ps-2 (1054) + ps+2 (1054) = 3162 entries

Why only pitch shift?
  - Tempo / time-stretch: invalidates "fast/slow tempo" captions.
  - Noise injection: invalidates "high/low quality" captions.
  - Pitch shift ±2 semitones: musically plausible; preserves all caption
    descriptions (instruments, mood, genre, quality).

Output
------
  musiccaps_audio/{ytid}_ps{:+d}.wav    augmented audio files
  data/train_augmented.json              original + augmented entries

Usage
-----
  python qwen/augment_dataset.py                         # default: ±2 semitones
  python qwen/augment_dataset.py --semitones -1 1 -2 2   # ±1 and ±2
  python qwen/augment_dataset.py --dry_run               # preview without saving
  python qwen/augment_dataset.py --workers 6             # more CPU parallelism
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR    = PROJECT_ROOT / "musiccaps_audio"
DATA_DIR     = PROJECT_ROOT / "data"
SR           = 16000


# ─── Core augmentation ───────────────────────────────────────────────────────

def pitch_shift_clip(src_path: Path, dst_path: Path, n_steps: int) -> bool:
    """
    Load audio, pitch-shift by n_steps semitones, save to dst_path.
    Returns True on success, False on error.
    """
    if dst_path.exists():
        return True  # already done, resume

    try:
        y, _ = librosa.load(str(src_path), sr=SR, mono=True)
        y_shifted = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
        sf.write(str(dst_path), y_shifted, SR, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"\n  [skip] {src_path.name}: {e}")
        return False


def augment_item(item: dict, n_steps: int, dry_run: bool) -> dict | None:
    """Return a new dataset entry for the pitch-shifted variant, or None on failure."""
    orig_path = PROJECT_ROOT / item["wav_path"].lstrip("./")
    ytid      = orig_path.stem
    sign      = f"{n_steps:+d}"                      # e.g. "+2" or "-2"
    aug_name  = f"{ytid}_ps{sign}.wav"
    aug_path  = AUDIO_DIR / aug_name

    if not dry_run:
        ok = pitch_shift_clip(orig_path, aug_path, n_steps)
        if not ok:
            return None

    return {
        "wav_path"            : f"./musiccaps_audio/{aug_name}",
        "prompt"              : item["prompt"],
        "ground_truth_caption": item["ground_truth_caption"],
        "aspect_list"         : item["aspect_list"],
        "_augmentation"       : f"pitch_shift_{sign}",
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=str(DATA_DIR / "train.json"),
        help="Source training manifest (default: data/train.json)",
    )
    parser.add_argument(
        "--output", default=str(DATA_DIR / "train_augmented.json"),
        help="Output manifest with original + augmented entries",
    )
    parser.add_argument(
        "--semitones", nargs="+", type=int, default=[-2, 2],
        help="Semitone shifts to apply (default: -2 2). "
             "Each value creates one variant per clip.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel CPU workers for pitch shifting (default: 4)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print plan without saving any files",
    )
    args = parser.parse_args()

    # Load original training data
    with open(args.input) as f:
        originals = json.load(f)

    n_orig   = len(originals)
    n_shifts = len(args.semitones)
    n_total  = n_orig * (1 + n_shifts)

    print(f"Input            : {args.input}  ({n_orig} clips)")
    print(f"Semitone shifts  : {args.semitones}  ({n_shifts} variants per clip)")
    print(f"Total output     : {n_total}  (original {n_orig} + augmented {n_orig*n_shifts})")
    print(f"Workers          : {args.workers}")
    print(f"Output           : {args.output}")
    if args.dry_run:
        print("\n[DRY RUN] No files will be created.\n")

    # Build work queue: (item, n_steps) for all augmentations
    tasks = [(item, n) for item in originals for n in args.semitones]

    augmented_records = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(augment_item, item, n_steps, args.dry_run): (item, n_steps)
            for item, n_steps in tasks
        }

        with tqdm(total=len(tasks), unit="clip",
                  desc="Augmenting", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    augmented_records.append(result)
                else:
                    failed += 1
                pbar.set_postfix(ok=len(augmented_records), fail=failed)
                pbar.update(1)

    # Combine: originals first, then augmented (keeps original indices stable)
    full_dataset = originals + augmented_records

    print(f"\nResults:")
    print(f"  Original clips      : {n_orig}")
    print(f"  Augmented generated : {len(augmented_records)}  (failed: {failed})")
    print(f"  Total entries       : {len(full_dataset)}")

    if not args.dry_run:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = Path(args.output).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(full_dataset, f, indent=4, ensure_ascii=False)
        tmp.replace(Path(args.output))
        print(f"\nSaved → {args.output}")
        print(f"\nTo use augmented data, update train_lora_qwen.py:")
        print(f'  DATA_JSON_TRAIN = os.path.join(PROJECT_ROOT, "data", "train_augmented.json")')
        print(f"  EXP_ID = \"exp-006\"")
    else:
        print(f"\n[DRY RUN] Would write {len(full_dataset)} entries to {args.output}")


if __name__ == "__main__":
    main()
