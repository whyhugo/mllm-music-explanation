"""
MusicCaps downloader — supports resume and parallel downloads.

Usage
-----
# Download up to 500 samples (skips already-downloaded ones)
python salmonn/download_musiccaps.py --target 500

# Download the entire dataset (~5521 clips, expect 60-75% success)
python salmonn/download_musiccaps.py --target all

# Adjust parallel workers (default 3; more → faster but risks rate-limiting)
python salmonn/download_musiccaps.py --target 500 --workers 4

# Force re-download even if file exists
python salmonn/download_musiccaps.py --target 500 --force

Output
------
musiccaps_audio/{ytid}.wav        — raw audio clips (16 kHz, mono)
musiccaps_processed.json          — dataset manifest (path, prompt, caption, aspects)
"""

import os
import ast
import json
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ── Paths (relative to project root = one level above this script) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR    = PROJECT_ROOT / "musiccaps_audio"
JSON_PATH    = PROJECT_ROOT / "musiccaps_processed.json"

MUSIC_PROMPT = "Please describe this music in detail and list its aspects."
YT_DLP       = [sys.executable, "-m", "yt_dlp"]   # always use venv's yt-dlp


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_existing() -> dict[str, dict]:
    """Return {ytid: record} for every entry already in the JSON manifest."""
    if not JSON_PATH.exists():
        return {}
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {Path(r["wav_path"]).stem: r for r in data}


def parse_aspect_list(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []


def download_clip(item: dict, audio_dir: Path, force: bool) -> dict | None:
    """
    Download a single MusicCaps clip.
    Returns the record dict on success, None on failure.
    """
    ytid    = item["ytid"]
    start_s = item["start_s"]
    end_s   = item["end_s"]
    out_path = audio_dir / f"{ytid}.wav"

    # Skip if already present (unless --force)
    if out_path.exists() and not force:
        return {
            "wav_path"            : f"./musiccaps_audio/{ytid}.wav",
            "prompt"              : MUSIC_PROMPT,
            "ground_truth_caption": item["caption"],
            "aspect_list"         : parse_aspect_list(item["aspect_list"]),
        }

    yt_url  = f"https://www.youtube.com/watch?v={ytid}"
    command = [
        *YT_DLP,
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",   # 16 kHz mono
        "--download-sections", f"*{start_s}-{end_s}",
        "--no-playlist",
        "-o", str(out_path),
        "--",
        yt_url,
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
        if not out_path.exists():
            return None   # yt-dlp exited 0 but produced no file
        return {
            "wav_path"            : f"./musiccaps_audio/{ytid}.wav",
            "prompt"              : MUSIC_PROMPT,
            "ground_truth_caption": item["caption"],
            "aspect_list"         : parse_aspect_list(item["aspect_list"]),
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def save_manifest(records: list[dict]) -> None:
    tmp = JSON_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp.replace(JSON_PATH)   # atomic write


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download MusicCaps audio clips.")
    parser.add_argument(
        "--target", default="500",
        help="Number of clips to have in total after download, or 'all' for the full dataset. "
             "Already-downloaded clips count toward this target. (default: 500)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel download workers. Keep ≤ 4 to avoid YouTube rate-limiting. (default: 3)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download clips that already exist on disk.",
    )
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load MusicCaps metadata
    print("Loading MusicCaps metadata from HuggingFace …")
    ds = load_dataset("google/MusicCaps", split="train")
    total_available = len(ds)
    print(f"  Total clips in dataset : {total_available}")

    # Determine target count
    if args.target.lower() == "all":
        target = total_available
    else:
        target = int(args.target)

    # Load existing downloaded records
    existing = load_existing()
    print(f"  Already downloaded     : {len(existing)}")
    print(f"  Target total           : {target}")
    needed = max(0, target - len(existing))
    print(f"  Need to download       : {needed}")

    if needed == 0:
        print("Target already reached. Use --force to re-download or increase --target.")
        return

    # Build work queue: exclude already-downloaded ytids
    queue = [
        ds[i] for i in range(total_available)
        if ds[i]["ytid"] not in existing
    ][:needed + int(needed * 0.30)]   # fetch 30% extra to compensate for failures

    print(f"\nStarting download  (workers={args.workers}, queue={len(queue)}) …\n")

    new_records: list[dict] = []
    failed = 0
    save_interval = 20   # save manifest every N completions

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, item, AUDIO_DIR, args.force): item["ytid"]
            for item in queue
        }

        with tqdm(total=min(needed, len(queue)), unit="clip", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                ytid   = futures[future]
                result = future.result()

                if result:
                    new_records.append(result)
                    pbar.set_postfix(ok=len(new_records), fail=failed)
                    pbar.update(1)

                    # Periodic save so progress isn't lost on interruption
                    if len(new_records) % save_interval == 0:
                        all_records = list(existing.values()) + new_records
                        save_manifest(all_records)

                    if len(new_records) >= needed:
                        # Cancel remaining futures once we have enough
                        for f in futures:
                            f.cancel()
                        break
                else:
                    failed += 1
                    pbar.set_postfix(ok=len(new_records), fail=failed)

    # Final save
    all_records = list(existing.values()) + new_records
    save_manifest(all_records)

    print(f"\n{'='*50}")
    print(f"  Downloaded successfully : {len(new_records)}")
    print(f"  Failed / unavailable   : {failed}")
    print(f"  Total in manifest      : {len(all_records)}")
    print(f"  Manifest saved to      : {JSON_PATH}")
    print(f"{'='*50}")

    if len(all_records) < target:
        shortfall = target - len(all_records)
        print(f"\n  Note: {shortfall} clips short of target (likely deleted/private YouTube videos).")
        print(f"  Re-run with --target {target + shortfall + 50} to try fetching more alternatives.")


if __name__ == "__main__":
    main()
