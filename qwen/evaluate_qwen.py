"""
Qwen2-Audio evaluation on MusicCaps
------------------------------------
Loads the fine-tuned LoRA checkpoint (or the base model for baseline),
runs inference on every entry in musiccaps_processed.json, and saves
a JSON with generated captions alongside ground-truth for metric scoring.

Checkpoint format support
--------------------------
This script supports two checkpoint formats:

1. PEFT format (produced by train_lora_qwen.py with SavePeftAdapterCallback):
   checkpoint-300/
   ├── adapter_config.json          ← PEFT adapter config
   ├── adapter_model.safetensors   ← LoRA weights only
   └── projector_weights.pt        ← multi_modal_projector weights

2. Trainer-only format (if callback was absent — full model state dict):
   checkpoint-300/
   ├── model.safetensors            ← or sharded model-xxxxx-of-xxxxx.safetensors
   └── model.safetensors.index.json ← present only for sharded models

Usage
-----
# Baseline (no LoRA — evaluate the base model)
python qwen/evaluate_qwen.py

# Fine-tuned checkpoint (run from project root)
python qwen/evaluate_qwen.py --lora_path outputs/qwen_musiccaps_finetuned/checkpoint-300
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# LoRA config must match train_lora_qwen.py exactly
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=None,
)

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SR      = 16000
MAX_AUDIO_SECS = 30
MUSIC_PROMPT   = "Please describe this music in detail and list its aspects."


# =====================================================================
# Audio helpers
# =====================================================================
def load_audio(wav_path: str) -> np.ndarray:
    abs_path = os.path.join(PROJECT_ROOT, wav_path.lstrip("./"))
    waveform, sr = torchaudio.load(abs_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    if waveform.shape[1] > MAX_AUDIO_SECS * TARGET_SR:
        waveform = waveform[:, : MAX_AUDIO_SECS * TARGET_SR]
    return waveform.squeeze(0).numpy()


def build_prompt_text(processor) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": MUSIC_PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


# =====================================================================
# LoRA checkpoint loading (supports both PEFT format & Trainer format)
# =====================================================================
def _load_state_dict_from_trainer_ckpt(ckpt_dir: str) -> dict:
    """Load model state dict from a Trainer checkpoint (handles sharding)."""
    from safetensors.torch import load_file

    index_path  = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_dir, "model.safetensors")
    pt_path     = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(index_path):
        print("  Detected sharded safetensors checkpoint.")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard in shard_files:
            state_dict.update(load_file(os.path.join(ckpt_dir, shard)))
        return state_dict

    if os.path.exists(single_path):
        print("  Detected single safetensors checkpoint.")
        return load_file(single_path)

    if os.path.exists(pt_path):
        print("  Detected PyTorch checkpoint.")
        return torch.load(pt_path, map_location="cpu")

    raise FileNotFoundError(
        f"No model weights found in {ckpt_dir}. "
        "Expected model.safetensors[.index.json] or pytorch_model.bin."
    )


def load_lora_checkpoint(model, lora_path: str):
    """
    Load LoRA (and optionally projector) weights into model.

    Handles two formats:
    - PEFT format  : directory contains adapter_config.json
    - Trainer format: directory contains full model.safetensors
    """
    lora_path = os.path.realpath(lora_path)   # resolve symlinks / relative paths

    adapter_cfg = os.path.join(lora_path, "adapter_config.json")

    # ------------------------------------------------------------------
    # Case 1: Proper PEFT checkpoint (saved by SavePeftAdapterCallback)
    # ------------------------------------------------------------------
    if os.path.exists(adapter_cfg):
        print(f"  Format: PEFT adapter  ({lora_path}/adapter_config.json found)")
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model,
            lora_path,
            is_trainable=False,
        )

        # Load projector weights if present
        proj_pt = os.path.join(lora_path, "projector_weights.pt")
        if os.path.exists(proj_pt) and hasattr(model.model, "multi_modal_projector"):
            proj_state = torch.load(proj_pt, map_location="cpu")
            model.model.multi_modal_projector.load_state_dict(proj_state, strict=True)
            print("  Loaded projector weights from projector_weights.pt")

        return model

    # ------------------------------------------------------------------
    # Case 2: Raw Trainer checkpoint (full model state dict)
    # Reconstruct the PEFT structure then extract only LoRA keys.
    # ------------------------------------------------------------------
    print(f"  Format: Trainer checkpoint  (no adapter_config.json, falling back)")
    print("  Reconstructing PEFT model from training config...")

    # Inject LoRA with the same config used during training
    model.model.language_model = get_peft_model(model.model.language_model, LORA_CONFIG)

    state_dict = _load_state_dict_from_trainer_ckpt(lora_path)

    # Extract LoRA adapter keys  (model.language_model.base_model.model.*.lora_*)
    lm_prefix = "model.language_model."
    lora_adapter_state = {
        k[len(lm_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(lm_prefix) and ("lora_A" in k or "lora_B" in k)
    }

    if not lora_adapter_state:
        print(
            "  WARNING: No LoRA keys found in checkpoint. "
            "The model will behave as the base model."
        )
    else:
        result = set_peft_model_state_dict(model.model.language_model, lora_adapter_state)
        n_loaded = len(lora_adapter_state)
        n_unexpected = len(result.unexpected_keys) if hasattr(result, "unexpected_keys") else 0
        print(f"  Loaded {n_loaded} LoRA tensors  ({n_unexpected} unexpected keys).")

    # Extract projector weights
    proj_prefix = "model.multi_modal_projector."
    proj_state = {
        k[len(proj_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(proj_prefix)
    }
    if proj_state and hasattr(model.model, "multi_modal_projector"):
        model.model.multi_modal_projector.load_state_dict(proj_state, strict=False)
        print(f"  Loaded {len(proj_state)} projector tensors from full checkpoint.")

    return model


# =====================================================================
# Inference
# =====================================================================
@torch.no_grad()
def generate_caption(model, processor, audio: np.ndarray, prompt_text: str) -> str:
    inputs = processor(
        text=[prompt_text],
        audio=[audio],
        return_tensors="pt",
        sampling_rate=TARGET_SR,
    )
    # Move to device only — do NOT cast dtype here (input_ids are int64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.1,
    )
    # Decode only newly generated tokens (skip the prompt)
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2-Audio on MusicCaps captioning task."
    )
    parser.add_argument(
        "--model_id", default="Qwen/Qwen2-Audio-7B-Instruct",
        help="HuggingFace model ID or local path to base model.",
    )
    parser.add_argument(
        "--lora_path", default=None,
        help=(
            "Path to a LoRA checkpoint directory (relative to CWD or absolute). "
            "Supports both PEFT format and raw Trainer checkpoint format. "
            "Omit to evaluate the base model without any fine-tuning."
        ),
    )
    parser.add_argument(
        "--data_json",
        default=os.path.join(PROJECT_ROOT, "musiccaps_processed.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "outputs", "qwen_validation_results.json"),
    )
    args = parser.parse_args()

    # Resolve lora_path relative to CWD (user runs from project root)
    if args.lora_path and not os.path.isabs(args.lora_path):
        args.lora_path = os.path.join(os.getcwd(), args.lora_path)

    print(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading model from: {args.model_id}")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.lora_path:
        if not os.path.exists(args.lora_path):
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {args.lora_path}\n"
                "Make sure training has completed and the path is correct."
            )
        print(f"Applying LoRA weights from: {args.lora_path}")
        model = load_lora_checkpoint(model, args.lora_path)
    else:
        print("No --lora_path provided. Evaluating base model (baseline).")

    model.eval()

    with open(args.data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    prompt_text = build_prompt_text(processor)
    results = []

    print(f"Running inference on {len(dataset)} samples...")
    for item in tqdm(dataset):
        try:
            audio = load_audio(item["wav_path"])
            generated = generate_caption(model, processor, audio, prompt_text)
            results.append(
                {
                    "wav_path"            : item["wav_path"],
                    "generated_text"      : generated,
                    "ground_truth_caption": item["ground_truth_caption"],
                    "aspect_list"         : item["aspect_list"],
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

    print(f"\nDone! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
