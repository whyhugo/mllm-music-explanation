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

Usage (run from project root)
-----------------------------
# Zero-shot baseline on the canonical held-out test split
python gemma/evaluate_gemma.py

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

# Generation hyper-parameters mirror evaluate_qwen.py:generate_caption so the
# two models are compared under the same decoding regime.
GEN_KWARGS = dict(
    max_new_tokens=256,
    num_beams=4,
    do_sample=False,
    repetition_penalty=1.1,
)


# =====================================================================
# Inference
# =====================================================================
@torch.no_grad()
def generate_caption(model, processor, abs_wav_path: str) -> str:
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
    generated = model.generate(**inputs, **GEN_KWARGS)

    # Decode only the newly generated tokens, then strip Gemma response markers.
    response = processor.decode(generated[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)
    # parse_response may return a string or a structured object depending on
    # the transformers version; normalise to a clean string.
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        return str(parsed.get("text", parsed)).strip()
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
    args = parser.parse_args()

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
            generated = generate_caption(model, processor, abs_path)
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
