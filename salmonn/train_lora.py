import os
import sys
import torch
import torchaudio
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Resolve paths relative to project root (one level up from salmonn/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SALMONN"))
from model import SALMONN

# =====================================================================
# 1. 載入模型與配置 LoRA
# =====================================================================
print("Loading model for Fine-tuning...")

SALMONN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SALMONN")

model = SALMONN(
    ckpt=os.path.join(SALMONN_DIR, "salmonn_v1.pth"),
    whisper_path="openai/whisper-large-v2",
    beats_path=os.path.join(SALMONN_DIR, "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
    vicuna_path="lmsys/vicuna-13b-v1.1",
    lora=False,
)

# 凍結所有基礎模型參數
for param in model.parameters():
    param.requires_grad = False

# 打開 Speech Q-Former Projector 的梯度
if hasattr(model, "speech_Qformer"):
    for param in model.speech_Qformer.parameters():
        param.requires_grad = True
        param.data = param.data.to(torch.float32)

# 針對 Vicuna LLM 加入 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model.llama_model = get_peft_model(model.llama_model, lora_config)
print("LoRA modules injected successfully.")

# =====================================================================
# 2. 讀取與處理 MusicCaps 資料集
# =====================================================================
print("Loading dataset...")
DATA_JSON = os.path.join(PROJECT_ROOT, "musiccaps_processed.json")
raw_dataset = load_dataset("json", data_files={"train": DATA_JSON}, split="train")

TARGET_SR = 16000
TARGET_LENGTH = 160000  # 16000Hz * 10s


def preprocess_function(examples):
    audio_tensors = []
    for wav_path in examples["wav_path"]:
        # Resolve relative path (e.g. ./musiccaps_audio/xxx.wav) from project root
        abs_path = os.path.join(PROJECT_ROOT, wav_path.lstrip("./"))
        waveform, sample_rate = torchaudio.load(abs_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)

        if waveform.shape[1] < TARGET_LENGTH:
            pad_size = TARGET_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[1] > TARGET_LENGTH:
            waveform = waveform[:, :TARGET_LENGTH]

        audio_tensors.append(waveform.squeeze(0))

    return {
        "audio": audio_tensors,
        "text": [
            f"USER: {p} ASSISTANT: {gt}"
            for p, gt in zip(examples["prompt"], examples["ground_truth_caption"])
        ],
    }


print("Preprocessing audio files...")
train_dataset = raw_dataset.map(preprocess_function, batched=True)


# =====================================================================
# 3. Data Collator
# =====================================================================
def custom_collate_fn(batch):
    audios = [item["audio"] for item in batch]
    texts = [item["text"] for item in batch]
    return {
        "audio": torch.stack(audios),
        "text": texts,
    }


# =====================================================================
# 4. 啟動訓練
# =====================================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "salmonn_musiccaps_finetuned")

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=custom_collate_fn,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        save_steps=500,
        logging_steps=50,
        remove_unused_columns=False,
    ),
)

import time
import json
start_time = time.time()

print("Starting LoRA training...")
trainer.train()

end_time = time.time()
elapsed_sec = end_time - start_time
elapsed_min = elapsed_sec / 60.0
elapsed_hr = elapsed_sec / 3600.0

# Calculate disk usage of checkpoints
total_bytes = 0
if os.path.exists(OUTPUT_DIR):
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if os.path.exists(fp) and not os.path.islink(fp):
                    total_bytes += os.path.getsize(fp)
            except Exception:
                pass
total_gb = total_bytes / (1024 ** 3)

print(f"\n{'='*65}")
print(f"Training complete. Checkpoints saved to: {OUTPUT_DIR}")
print(f"Training Runtime: {elapsed_sec:.2f} seconds ({elapsed_min:.2f} minutes / {elapsed_hr:.2f} hours)")
print(f"Total storage space used by outputs: {total_gb:.4f} GB")
print(f"{'='*65}\n")


