# mllm-music

以 LoRA 微調 **Qwen2-Audio-7B-Instruct**，執行自動音樂描述生成（Music Captioning），
基準資料集為 [MusicCaps](https://huggingface.co/datasets/google/MusicCaps)（Google DeepMind）。

---

## 目錄

1. [模型架構與訓練策略](#模型架構與訓練策略)
2. [資料集與切分標準](#資料集與切分標準)
3. [實驗規範](#實驗規範)
4. [誠實評估結果](#誠實評估結果)
5. [環境建置](#環境建置)
6. [執行流程](#執行流程)
7. [超參數說明](#超參數說明)
8. [已知陷阱與教訓](#已知陷阱與教訓)
9. [專案結構](#專案結構)

---

## 模型架構與訓練策略

```
Audio (.wav, 16 kHz)
       │
       ▼
 ┌─────────────┐     ┌──────────────────────┐
 │ Audio Tower │────▶│ Multi-Modal Projector │ ← 解凍（可訓練）
 │  (Whisper)  │     └──────────────────────┘
 └─────────────┘               │
  （凍結）                      ▼
                      ┌────────────────┐
                      │  Qwen2-7B LLM  │ ← LoRA on q/k/v/o_proj
                      └────────────────┘
                               │
                               ▼
                      "A mellow piano ballad
                       with soft female vocals..."
```

**訓練策略（固定，請勿變更）**

| 組件 | 狀態 | 說明 |
|------|------|------|
| Audio Tower (Whisper) | 凍結 | 不更新梯度 |
| Multi-Modal Projector | 解凍 | 橋接音頻特徵與 LLM 輸入空間 |
| Qwen2-7B LLM | LoRA | r=16, alpha=32, dropout=0.05 |
| LoRA 目標模組 | q/k/v/o_proj | LLM 自注意力層 |
| 可訓練參數 | ~16.8 M / 7.77 B | 佔總參數 0.22% |

---

## 資料集與切分標準

> **所有人必須使用同一套切分，才能確保實驗結果可比較。**

### 來源

- **原始資料集**：[google/MusicCaps](https://huggingface.co/datasets/google/MusicCaps)（5521 筆，包含 YouTube 音頻連結）
- **實際可用音頻**：受 YouTube 下載成功率限制，目前 **1267 筆**（約 23%，含歷次下載累積）
- **主資料清單**：`musiccaps_processed.json`（根目錄），記錄所有已下載的音頻路徑與標注

### 標準切分方式（唯一認可）

使用 **`data/split.py`** 生成三分切割。這是目前最完整的切分腳本，採用音樂面向標籤（aspect tag）做分層抽樣，確保 train / val / test 三組在音樂類型分布上相似。

```
musiccaps_processed.json  →  data/split.py  →  data/musiccaps_train.json
                                               data/musiccaps_val.json
                                               data/musiccaps_test.json
```

**切分比例與參數（固定不變）**

```bash
python data/split.py  # 預設：80/10/10，seed=42，分層依 top-15 aspect tags
```

| 檔案 | 比例 | 筆數 | 用途 |
|------|------|------|------|
| `data/musiccaps_train.json` | 80% | **1011** | 訓練 |
| `data/musiccaps_val.json` | 10% | **128** | 驗證 / 早停 / checkpoint 選擇 |
| `data/musiccaps_test.json` | 10% | **128** | **最終評估，訓練期間禁止觸碰** |

> 上述切分以 1267 筆可用音頻、seed=42、aspect-stratified 生成，已 commit 至 repo。
> **不需要自行重跑 split.py**；只有資料集更新後才需重新生成並重新 commit。

```bash
# 查看切分後各組的 aspect 分布
python data/split.py --stats

# 自訂比例（ablation 用，需在 runs.json 中記錄差異）
python data/split.py --val_ratio 0.15 --test_ratio 0.15 --seed 42
```

### 舊有檔案說明

以下檔案已從 repo 移除，**不應再出現或重新建立**：

| 已刪除的舊檔案 | 原出處 | 移除原因 |
|----------------|--------|----------|
| `data/train.json` | `qwen/split_dataset.py`（90/10） | 無 test set，切分策略不同 |
| `data/val.json` | 同上 | 同上 |
| `data/train_augmented.json` | `qwen/augment_dataset.py` | exp-006 已證實 pitch augmentation 無效 |
| `qwen/split_dataset.py` | 舊切分腳本 | 以 `data/split.py` 取代 |

exp-005b / exp-006 的結果使用上述舊切分，若要重現那些數字需手動還原（見 git log）。

### 切分不得更動的原因

一旦切分固定並提交，後續所有實驗的 val / test 評估才能直接比較。更改種子或比例，
代表對應的 val/test 結果需要重新與 baseline 比較，**無法與現有實驗數字並排**。

---

## 實驗規範

### 命名規則

```
exp-{三位數}           exp-007
exp-{三位數}-{說明}    exp-007-lora-r32
```

在 `qwen/train_lora_qwen.py` 頂部設定 `EXP_ID`，此值用於 TensorBoard run 命名與 checkpoint 子資料夾。

### 每次實驗前必做

1. 在 `experiments/runs.json` 新增一筆 entry，記錄計畫中的超參數與資料集。
2. 確認 `EXP_ID` 已更新為新的 ID。
3. **確認使用 `data/musiccaps_train.json` 訓練，`data/musiccaps_val.json` 做 val**。
4. 確認 `save_total_limit=None` 或足夠大，避免 best checkpoint 被覆寫。

### 每次實驗後必做

1. 在 `experiments/runs.json` 補上實際訓練結果（val_loss 曲線、best checkpoint、評估指標）。
2. 提交 `experiments/runs.json` 更新（見格式說明）。
3. 將 val 集評估指標與 `baseline-val117`（或對應 baseline）做對比，記錄相對提升百分比。

### runs.json 必填欄位

```json
{
  "id": "exp-007",
  "date": "YYYY-MM-DD",
  "name": "...",
  "status": "planned | running | completed | invalid",
  "description": "這次實驗要驗證什麼假設",
  "dataset": {
    "train": "data/musiccaps_train.json (N samples)",
    "val":   "data/musiccaps_val.json  (N samples, held-out)"
  },
  "training": {
    "epochs": ...,
    "learning_rate": ...,
    "max_grad_norm": ...,
    "eval_strategy": "epoch",
    "save_total_limit": null
  },
  "val_loss_history": [...],
  "best_checkpoint": { "epoch": ..., "val_loss": ... },
  "evaluation": {
    "eval_set": "data/musiccaps_val.json",
    "baseline": { "ROUGE-1": ..., "ROUGE-2": ..., "ROUGE-L": ..., "BLEU": ..., "METEOR": ... },
    "finetuned": { ... },
    "improvement_pct": { ... },
    "avg_improvement_vs_baseline": ...
  },
  "observations": ["..."],
  "next_experiment": "exp-008"
}
```

### 公平比較的原則

| 規則 | 原因 |
|------|------|
| 評估永遠在 val set（訓練中未見過的樣本）進行 | exp-004 在 in-sample 得到 +55%，真實數字是 +24%（exp-005b） |
| baseline 和 finetuned 用**同一批** val 樣本 | baseline-val117 是 exp-005b/006 的正確對照 |
| test set 僅在論文 / 最終報告時使用一次 | 多次在 test set 上調整等同 test set 洩漏 |
| 記錄 val_loss 每個 epoch 的數值 | 才能確認 overfitting 起點 |
| `save_total_limit=None` + `BestValCheckpointCallback` | exp-005 因 limit=3 丟失最佳 checkpoint |

---

## 誠實評估結果

> 下表僅列出使用**真正 held-out val set** 的實驗。
> exp-001～exp-004 的指標在 in-sample 或未完整切分的資料上計算，**不應作為比較基準**。

**Baseline**（無微調，`Qwen2-Audio-7B-Instruct` 原始模型，在 data/val.json 117 筆）

| ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|---------|---------|---------|------|--------|
| 0.5114 | 0.2740 | 0.4063 | 0.2285 | 0.4128 |

**微調實驗**（eval set = data/val.json，117 筆 held-out）

| ID | 訓練集 | 訓練樣本 | Epochs | Best val epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | Avg Δ |
|----|--------|----------|--------|----------------|---------|---------|---------|------|--------|-------|
| exp-005b | train.json | 1054 | 5（ep3 最佳） | ep3 / val_loss=1.1125 | 0.5875 | 0.3601 | 0.4828 | 0.3066 | 0.4924 | **+24.0%** |
| exp-006 | train_augmented.json | 3162 | 15（ep2 最佳） | ep2 / val_loss=1.0937 | 0.5408 | 0.3133 | 0.4393 | 0.2659 | 0.4496 | +10.7% |

**關鍵發現**

- **exp-005b 是目前最佳誠實結果**：+24% 平均提升（BLEU +37.7%，ROUGE-2 +32.2%）。
- **Pitch augmentation 無效**（exp-006）：相同標注重複 3 次讓模型快速記憶 1054 個片段的分布，overfitting 更早發生（ep2 vs ep4），最終比 exp-005b 差 -10.4%。
- **資料多樣性 > 資料數量**：augmented 3162 筆（含重複）不如 1054 筆真實多樣樣本。
- **Overfitting 點**：1054 筆訓練資料在 epoch 3 飽和；推測需 ~3000 筆多樣資料才能將最佳點推到 ep10。

**探索性實驗（不可直接與上表比較）**

| ID | 評估集 | 樣本 | 問題 |
|----|--------|------|------|
| exp-001 | 訓練集本身（93） | 93 | in-sample |
| exp-002 | 訓練集本身（93） | 93 | in-sample |
| exp-003 | 額外 500 筆（與訓練集高度重疊） | 500 | 部分 in-sample |
| exp-004 | 所有 1003 筆（訓練集 = 評估集） | 1000 | 完全 in-sample，+55.65% 為膨脹數字 |

---

## 環境建置

### 步驟 1：安裝套件

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r qwen/requirements.txt
```

### 步驟 2：取得音頻（二擇一）

**選項 A — 使用共享音頻目錄（推薦，組織內部成員）**

音頻已由管理員統一下載並放置於共享路徑。建立 symlink 指向該目錄即可：

```bash
ln -s /path/to/shared/musiccaps_audio ./musiccaps_audio
```

確認可用：

```bash
ls musiccaps_audio/ | wc -l   # 應顯示 1267
```

**選項 B — 自行從 YouTube 下載（首次建置或新環境）**

```bash
python salmonn/download_musiccaps.py --target all --workers 4
# 耗時約 2 小時；受 YouTube 版權限制，可下載上限約 1267 筆
```

### 步驟 3：確認切分檔案

切分 JSON 已 commit 至 repo，**clone 後直接可用，不需重跑 split.py**：

```
data/musiccaps_train.json  — 1011 筆（訓練）
data/musiccaps_val.json    —  128 筆（驗證）
data/musiccaps_test.json   —  128 筆（最終評估，訓練期間禁止使用）
```

---

**硬體需求**

| 項目 | 最低要求 |
|------|----------|
| GPU VRAM | 24 GB（已在 RTX 3090 / 4090 測試） |
| CUDA | 11.8+ |
| Python | 3.10+ |
| 磁碟空間 | ~20 GB（音頻 ~920 MB + checkpoint 最多 ~16 GB） |

---

## 執行流程

### 訓練

```bash
# 在 qwen/train_lora_qwen.py 頂部設定 EXP_ID = "exp-007"（每次實驗前更換）

python qwen/train_lora_qwen.py

# 另開終端機監控訓練
bash qwen/launch_tb.sh
# 開啟瀏覽器：http://localhost:6006
```

**TensorBoard 重要指標**

| 指標 | 意義 |
|------|------|
| `eval/loss` | **主要早停依據**，應在最低點 checkpoint |
| `train/loss` | 訓練損失，應持續下降 |
| `train/lora_B_rms` | LoRA 適應強度，從 0 開始上升 |
| `train/grad_norm` | 梯度裁切前的範數（正常範圍：15–40） |
| `train/label_ratio` | 有效 token 佔比（正常範圍：0.07–0.25） |

### 評估

```bash
# 1. 評估 baseline（無 LoRA）
python qwen/evaluate_qwen.py \
    --input data/musiccaps_val.json \
    --output outputs/eval_baseline_valXX.json

# 2. 評估微調 checkpoint（從 best checkpoint 讀取）
python qwen/evaluate_qwen.py \
    --lora_path outputs/qwen_musiccaps_finetuned/checkpoint-best \
    --input data/musiccaps_val.json \
    --output outputs/eval_exp007.json

# 3. 計算指標並比較
python run_eval_compare.py \
    --baseline outputs/eval_baseline_valXX.json \
    --ours outputs/eval_exp007.json

# 4. 查看所有實驗摘要
python experiments/compare.py
python experiments/compare.py --detail exp-007
```

> **Test set 只在最終報告時執行一次**，將 `--input` 換成 `data/musiccaps_test.json`。

### 資料擴增（如需嘗試）

```bash
# 生成 pitch-shifted 變體（請先閱讀 exp-006 教訓）
python qwen/augment_dataset.py

# 查看說明
python qwen/augment_dataset.py --help
```

---

## 超參數說明

以下為目前訓練腳本（`qwen/train_lora_qwen.py`）的預設值，調整前請在 runs.json 說明原因。

| 超參數 | 預設值 | 說明 |
|--------|--------|------|
| `num_train_epochs` | 15 | 依 val_loss 早停，實際最佳通常在 ep2–ep4 |
| `learning_rate` | 1e-5 | exp-001 用 2e-5 導致梯度爆炸，降至此值穩定 |
| `gradient_accumulation_steps` | 8 | effective batch size = 8（batch=1 × accum=8） |
| `max_grad_norm` | 0.5 | 訓練中實際梯度範數 15–40（此為裁切後上限） |
| `warmup_steps` | 動態 | 設為 total_steps 的 10%，由腳本計算 |
| `lr_scheduler_type` | cosine | |
| `fp16` | True | |
| `eval_strategy` | epoch | 每個 epoch 在 val set 評估一次 |
| `save_total_limit` | None | **必須為 None**，否則 best checkpoint 可能被覆寫 |
| LoRA `r` | 16 | |
| LoRA `alpha` | 32 | alpha = 2r，標準設定 |
| LoRA `dropout` | 0.05 | |

---

## 已知陷阱與教訓

### 1. Audio 處理 kwarg 變更（exp-000 → exp-001）

transformers 5.x 將 `processor(..., audios=waveforms)` 改為 `audio=`（單數）。
使用舊 API 時模型訓練的是純文字，音頻特徵完全未被使用，但訓練 loss 仍下降。
→ **確認 `qwen/train_lora_qwen.py` 中 collator 使用 `audio=`（單數）**。

### 2. In-sample 評估導致指標膨脹（exp-004）

exp-004 在訓練集上評估得到 +55.65%；exp-005b 在 held-out val 得到 +24.02%。
兩者的差距（~25%）量化了 exp-004 的過擬合程度。
→ **評估必須在訓練時未見過的樣本上進行**。

### 3. Best checkpoint 被覆寫（exp-005）

`save_total_limit=3` 加上 `BestValCheckpointCallback` 導致 best checkpoint（ep3）被 ep5/14/15 覆寫，
實驗必須重跑（exp-005b）。
→ **設定 `save_total_limit=None`，`BestValCheckpointCallback` 必須掛載**。

### 4. Pitch augmentation 無效（exp-006）

3 倍 pitch-shifted 資料讓訓練集從 1054 增至 3162，但標注完全相同（音樂描述與音高無關），
模型更快記住 1054 個獨立片段的分布，overfitting 從 ep4 提前至 ep2。
最終 exp-006 比 exp-005b 差 -10.4%。
→ **同標注的 augmentation 增加重複，不增加多樣性**。

### 5. OOM 解法（確認已套用）

訓練時在 `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` 開頭設定，
eval 時使用 `eval_accumulation_steps` 與 `ClearCacheBeforeEvalCallback`。
→ 若遇到 OOM，先確認上述設定，再考慮降低 `per_device_eval_batch_size`。

### 6. Warmup 設定（exp-003 → exp-004）

exp-003 設 `warmup_steps=6`，佔 315 steps 的 1.9%（過短）。
正確設定應為 total_steps 的 10%（315 steps → warmup=31）。
→ 訓練腳本已改為動態計算，請確認沿用。

---

## 專案結構

```
mllm-music/
├── data/
│   ├── split.py                  # ← 標準切分腳本（80/10/10，seed=42）
│   ├── musiccaps_train.json      # ← 訓練集（標準，由 split.py 生成）
│   ├── musiccaps_val.json        # ← 驗證集（標準，由 split.py 生成）
│   └── musiccaps_test.json       # ← 測試集（最終評估才用，由 split.py 生成）
│
├── qwen/                         # Qwen2-Audio 微調 pipeline（主要研究）
│   ├── train_lora_qwen.py        # 訓練腳本（修改 EXP_ID 後執行）
│   ├── evaluate_qwen.py          # 推論 + 輸出 prediction JSON
│   ├── augment_dataset.py        # Pitch shift 資料擴增（供參考，exp-006 已證實無效）
│   ├── launch_tb.sh              # TensorBoard 啟動腳本
│   └── requirements.txt
│
├── salmonn/                      # SALMONN baseline（保留備查）
│   ├── SALMONN/                  # 模型代碼（BEATs, QFormer）
│   ├── train_lora.py
│   ├── evaluate_salmonn.py
│   ├── download_musiccaps.py     # MusicCaps 音頻下載器
│   └── requirements.txt
│
├── experiments/
│   ├── runs.json                 # 所有實驗的完整紀錄（必須維護）
│   └── compare.py                # CLI 比較表格
│
├── musiccaps_processed.json      # 主資料清單（音頻路徑 + 標注）
├── musiccaps_audio/              # 已下載的 .wav 音頻（~1171 筆）
├── run_eval_compare.py           # 多輪評估比較腳本（含 sample-level std）
└── outputs/                      # checkpoint + eval 結果（git ignored）
```

---

## 參考資料

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) — Alibaba DAMO Academy
- [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) — Google DeepMind
- [SALMONN](https://github.com/bytedance/SALMONN) — ByteDance Research
- [PEFT / LoRA](https://github.com/huggingface/peft) — Hugging Face
