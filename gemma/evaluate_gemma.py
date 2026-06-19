"""
Gemma 4 E4B (audio) zero-shot evaluation on MusicCaps
------------------------------------------------------
Research baseline: runs the instruction-tuned Gemma 4 E4B multimodal model
(google/gemma-4-E4B-it) on the MusicCaps captioning task WITHOUT any
fine-tuning, and saves a JSON in the exact same schema as
qwen/evaluate_qwen.py so the results are directly comparable to the
Qwen2-Audio experiments (run_eval_compare.py / compute_metrics_with_std.py).

Output schema (one object per clip), identical to evaluate_qwen.py:
    {
      "wav_path"            : "./musiccaps_audio/<id>.wav",
      "generated_text"      : "<model caption>",
      "ground_truth_caption": "<reference>",
      "aspect_list"         : [ ... ]
    }

Why a separate script (vs reusing evaluate_qwen.py)
---------------------------------------------------
Gemma 4 uses a different model class and a different multimodal API than
Qwen2-Audio:
  - class    : AutoModelForMultimodalLM      (Qwen: Qwen2AudioForConditionalGeneration)
  - audio in : passed inline in the chat message as {"type": "audio", "audio": path}
               and tokenized via apply_chat_template(..., tokenize=True)
  - decoding : processor.parse_response() strips Gemma's response markers
Requires transformers >= 5.5.0 (Gemma 4 support). The fairness-critical
bits — the prompt and the generation hyper-parameters — are kept identical
to the Qwen evaluation.

This same script also serves the dense Gemma 4 12B variant
(google/gemma-4-12B-it), which is encoder-free: it projects the raw audio
waveform directly into the LLM embedding space (no separate audio encoder).
Its multimodal API is identical to E4B, so only --model_id changes.

Usage (run from project root)
-----------------------------
# E4B zero-shot baseline on the canonical held-out test split (exp-007)
python gemma/evaluate_gemma.py

# Gemma 4 12B (encoder-free, raw-waveform) variant (exp-008).
# 12B is dense ~12B params (~24GB bf16); on a 24GB GPU drop to --num_beams 1
# to avoid OOM from beam search.
python gemma/evaluate_gemma.py \
    --model_id google/gemma-4-12B-it \
    --num_beams 1 \
    --output outputs/eval_gemma4_12b_baseline_test.json

# Custom split / output
python gemma/evaluate_gemma.py \
    --data_json data/musiccaps_val.json \
    --output outputs/eval_gemma4_baseline_val.json
"""

import os
import json
import argparse
import torch
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForMultimodalLM

# =====================================================================
# Constants — kept identical to qwen/evaluate_qwen.py for a fair compare
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
MUSIC_PROMPT = "Please describe this music in detail and list its aspects."

# Default generation hyper-parameters mirror evaluate_qwen.py:generate_caption
# so the two model families are compared under the same decoding regime.
# Overridable via CLI (--num_beams / --max_new_tokens) — useful for the dense
# 12B variant, where num_beams=4 may OOM on a 24 GB GPU (drop to --num_beams 1).
DEFAULT_NUM_BEAMS = 4
DEFAULT_MAX_NEW_TOKENS = 256


# =====================================================================
# Inference
# =====================================================================
@torch.no_grad()
def generate_caption(model, processor, abs_wav_path: str, gen_kwargs: dict) -> str:
    """Run Gemma 4 on one audio clip and return the cleaned caption text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": abs_wav_path},
                {"type": "text", "text": MUSIC_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    generated = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens, then strip Gemma response markers.
    response = processor.decode(generated[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)
    return _to_text(parsed)


def _to_text(parsed) -> str:
    """
    Normalise processor.parse_response() output to a plain caption string.

    Gemma 4's parse_response returns a chat-message dict
    {"role": "assistant", "content": "<caption>"} (and may return a list of
    such messages). Pull out the assistant content; fall back gracefully.
    """
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        content = parsed.get("content", parsed.get("text", parsed))
        return _to_text(content)
    if isinstance(parsed, (list, tuple)):
        return " ".join(_to_text(p) for p in parsed).strip()
    return str(parsed).strip()


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Gemma 4 E4B baseline on MusicCaps captioning."
    )
    parser.add_argument(
        "--model_id", default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID or local path (default: google/gemma-4-E4B-it).",
    )
    parser.add_argument(
        "--data_json",
        default=os.path.join(PROJECT_ROOT, "data", "musiccaps_test.json"),
        help="Evaluation split JSON (default: canonical held-out test set).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "outputs", "eval_gemma4_baseline_test.json"),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of samples (for a quick smoke test).",
    )
    parser.add_argument(
        "--num_beams", type=int, default=DEFAULT_NUM_BEAMS,
        help="Beam-search width (default 4, matching the Qwen eval). "
             "Set to 1 for the 12B variant if beam search OOMs on a 24 GB GPU.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Max generated tokens per caption (default 256).",
    )
    args = parser.parse_args()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
        repetition_penalty=1.1,
    )
    print(f"Decoding: {gen_kwargs}")

    print(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading model from: {args.model_id}")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    with open(args.data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]

    results = []
    print(f"Running zero-shot inference on {len(dataset)} samples...")
    for item in tqdm(dataset):
        abs_path = os.path.join(PROJECT_ROOT, item["wav_path"].lstrip("./"))
        if not os.path.exists(abs_path):
            print(f"[skip] missing audio: {abs_path}")
            continue
        try:
            generated = generate_caption(model, processor, abs_path, gen_kwargs)
            results.append(
                {
                    "wav_path"            : item["wav_path"],
                    "generated_text"      : generated,
                    "ground_truth_caption": item["ground_truth_caption"],
                    "aspect_list"         : item.get("aspect_list", []),
                }
            )
        except Exception as e:
            print(f"Error on {item['wav_path']}: {e}")

        torch.cuda.empty_cache()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nDone! {len(results)} captions saved to: {args.output}")
    print("Score with:")
    print(f"  python compute_metrics_with_std.py "
          f"--baseline outputs/eval_baseline_test.json --finetuned {args.output}")


if __name__ == "__main__":
    main()
