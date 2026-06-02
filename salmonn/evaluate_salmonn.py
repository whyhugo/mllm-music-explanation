import os
import sys
import json
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SALMONN"))
from model import SALMONN

# 1. 初始化模型
print("Loading SALMONN model...")
model = SALMONN.from_config("configs/salmonn_eval_config.yaml")
model.eval()
model.cuda()

# 2. 讀取資料
DATA_JSON = os.path.join(PROJECT_ROOT, "musiccaps_processed.json")
with open(DATA_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

# 3. 批次推論
print("Starting inference...")
with torch.no_grad():
    for data in tqdm(dataset):
        wav_path = data["wav_path"]
        # Resolve relative path from project root
        abs_wav_path = os.path.join(PROJECT_ROOT, wav_path.lstrip("./"))
        prompt = data["prompt"]

        try:
            samples = {"wav_path": abs_wav_path, "prompt": prompt}
            response = model.generate(samples)
            results.append({
                "wav_path": wav_path,
                "generated_text": response[0] if isinstance(response, list) else response,
                "ground_truth_caption": data["ground_truth_caption"],
                "aspect_list": data["aspect_list"],
            })
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

        torch.cuda.empty_cache()

# 4. 儲存結果
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "salmonn_validation_results.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Inference complete! Results saved to {OUTPUT_PATH}")
