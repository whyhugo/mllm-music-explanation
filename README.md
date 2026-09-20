# mllm-music-explanation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-%E2%89%A54.45-FFD21E)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-%E2%89%A50.12-FFD21E)](https://github.com/huggingface/peft)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-%E2%89%A52.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![librosa](https://img.shields.io/badge/librosa-%E2%89%A50.10-4B0082)](https://librosa.org/)
[![Qwen2-Audio](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen2--Audio--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
[![MusicCaps](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MusicCaps-blue)](https://huggingface.co/datasets/google/MusicCaps)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

以 LoRA 微調 **Qwen2-Audio-7B-Instruct**，執行自動音樂描述生成（Music Captioning），
基準資料集為 [MusicCaps](https://huggingface.co/datasets/google/MusicCaps)（Google DeepMind）。
另包含 **Gemma 4**（12B encoder-free / E4B encoder-based）zero-shot 跨模型對照，
以及 **Audio-TAM** 可解釋性分析（見 `report/`）。

---

## 目錄

1. [模型架構與訓練策略](#模型架構與訓練策略)
2. [資料集與切分標準](#資料集與切分標準)
3. [實驗規範](#實驗規範)
4. [誠實評估結果](#誠實評估結果)
5. [跨模型 Zero-Shot 對照（Gemma 4）](#跨模型-zero-shot-對照gemma-4)
6. [可解釋性分析（Audio-TAM）](#可解釋性分析audio-tam)
7. [環境建置](#環境建置)
8. [執行流程](#執行流程)
9. [超參數說明](#超參數說明)
10. [已知陷阱與教訓](#已知陷阱與教訓)
11. [論文 / 報告](#論文--報告)
12. [專案結構](#專案結構)

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
- **實際可用音頻**：受 YouTube 下載成功率限制，目前 **5165 筆**（約 94%，含歷次下載累積）
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
| `data/musiccaps_train.json` | 80% | **4125** | 訓練 |
| `data/musiccaps_val.json` | 10% | **520** | 驗證 / 早停 / checkpoint 選擇 |
| `data/musiccaps_test.json` | 10% | **520** | **最終評估，訓練期間禁止觸碰** |

> 上述切分以 5165 筆可用音頻、seed=42、aspect-stratified 生成，已 commit 至 repo。
> **不需要自行重跑 split.py**；只有資料集更新後才需重新生成並重新 commit。

```bash
# 查看切分後各組的 aspect 分布
python data/split.py --stats

# 自訂比例（ablation 用，需在 runs.json 中記錄差異）
python data/split.py --val_ratio 0.15 --test_ratio 0.15 --seed 42
```

### 早期（1267 筆時代）的切分

exp-005b / exp-006 是在資料集尚只有 1267 筆音頻時跑的，當時的切分為 `data/train.json`（1054）
/ `data/val.json`（117），這些檔案已從 repo 移除。**那批數字不可與目前 5165 筆切分的結果並排比較**，
若要重現需手動還原（見 git log）。

| 已刪除的舊檔案 | 原出處 | 移除原因 |
|----------------|--------|----------|
| `data/train.json` | `qwen/split_dataset.py`（90/10） | 無 test set，切分策略不同 |
| `data/val.json` | 同上 | 同上 |
| `data/train_augmented.json` | `qwen/augment_dataset.py` | exp-006 已證實 pitch augmentation 無效 |
| `qwen/split_dataset.py` | 舊切分腳本 | 以 `data/split.py` 取代 |

### 輔助 / 過渡期檔案（非標準切分）

| 檔案 | 說明 |
|------|------|
| `data/test.json`（465 筆） | 早期臨時測試集，**不作為評估基準** |
| `data/musiccaps_test_1001_1100.json`（100 筆） | 由 `data/create_test_set.py` 依索引區間切出的小型 smoke-test 子集 |

### 切分不得更動的原因

一旦切分固定並提交，後續所有實驗的 val / test 評估才能直接比較。更改種子或比例，
代表對應的 val/test 結果需要重新與 baseline 比較，**無法與現有實驗數字並排**。

---

## 實驗規範

### 命名規則

```
exp-{三位數}           exp-009
exp-{三位數}-{說明}    exp-009-lora-r32
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
  "id": "exp-009",
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
    "save_total_limit": null,
    "runtime_hours": ...,         // 實際訓練耗時（小時）
    "storage_gb": ...             // 訓練產出/Checkpoints 所佔用的硬碟空間（GB）
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
  "next_experiment": "exp-010"
}
```

### 公平比較的原則

| 規則 | 原因 |
|------|------|
| 評估永遠在 held-out 集（訓練中未見過的樣本）進行 | exp-004 在 in-sample 得到 +55%，同期 held-out 真實數字是 +24%（exp-005b） |
| baseline 和 finetuned 用**同一批** val 樣本 | baseline-val117 是 exp-005b/006 的正確對照 |
| test set 僅在論文 / 最終報告時使用一次 | 多次在 test set 上調整等同 test set 洩漏 |
| 記錄 val_loss 每個 epoch 的數值 | 才能確認 overfitting 起點 |
| `save_total_limit=None` + `BestValCheckpointCallback` | exp-005 因 limit=3 丟失最佳 checkpoint |

---

## 誠實評估結果

> 下表僅列出使用**真正 held-out 資料**的實驗。
> exp-001～exp-004 的指標在 in-sample 或未完整切分的資料上計算，**不應作為比較基準**。

### 最終結果（held-out test set，520 筆；論文採用）

全量訓練（4125 train / 520 val，15 epochs，best = ep3，val_loss = 1.0890，約 15 小時）後，
在 `data/musiccaps_test.json` 上做**唯一一次**評估。指標以「每個 clip 分數」的
Mean ± Sample SD 呈現，並以 1000 輪 Bootstrap 估算平均值的標準誤（SE）。

| 指標 | Baseline (M±SD) | Base SE | Fine-Tuned (M±SD) | FT SE | 相對提升 |
|------|-----------------|---------|-------------------|-------|----------|
| ROUGE-1 | 0.4826 ± 0.1987 | 0.0088 | **0.5035 ± 0.2080** | 0.0093 | +4.33% |
| ROUGE-2 | 0.2437 ± 0.2173 | 0.0096 | **0.2711 ± 0.2355** | 0.0105 | +11.24% |
| ROUGE-L | 0.3882 ± 0.2117 | 0.0093 | **0.4095 ± 0.2228** | 0.0099 | +5.50% |
| BLEU | 0.1798 ± 0.2023 | 0.0089 | **0.2044 ± 0.2223** | 0.0098 | +13.70% |
| METEOR | 0.4001 ± 0.2148 | 0.0095 | **0.4184 ± 0.2297** | 0.0102 | +4.58% |

**全量訓練的 loss 動態**（詳見 `report/main.tex` Experiment 3、`report/loss_curve.png`）

- val_loss 最低點在 **ep3（1.0890）**，之後在 1.09–1.15 之間震盪，**未出現 1267 筆時代那種單調發散**
  → 較大且分層的訓練集本身即具正則化效果。
- ep7 與 ep12 出現訓練 loss 突刺（6.17 → 9.04、7.52 → 9.29），推測與 effective batch size = 8 的梯度變異有關。

### 早期小規模實驗（1267 筆時代，**不可與上表並排比較**）

**Baseline**（無微調，`Qwen2-Audio-7B-Instruct` 原始模型，在舊 `data/val.json` 117 筆）

| ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|---------|---------|---------|------|--------|
| 0.5114 | 0.2740 | 0.4063 | 0.2285 | 0.4128 |

**微調實驗**（eval set = 舊 `data/val.json`，117 筆 held-out；corpus-level，無 bootstrap）

| ID | 訓練集 | 訓練樣本 | Epochs | Best val epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | Avg Δ |
|----|--------|----------|--------|----------------|---------|---------|---------|------|--------|-------|
| exp-005b | train.json | 1054 | 5（ep3 最佳） | ep3 / val_loss=1.1125 | 0.5875 | 0.3601 | 0.4828 | 0.3066 | 0.4924 | **+24.0%** |
| exp-006 | train_augmented.json | 3162 | 15（ep2 最佳） | ep2 / val_loss=1.0937 | 0.5408 | 0.3133 | 0.4393 | 0.2659 | 0.4496 | +10.7% |

> 這兩筆的 +24% / +10.7% 是 **corpus-level、117 筆小 val 集**的數字，樣本少且指標計算方式與最終 test
> 評估不同，因此明顯高於最終 test 上的 +4~14%。**對外引用一律以上方 test set 表格為準。**

**關鍵發現**

- **Pitch augmentation 無效**（exp-006）：相同標注重複 3 次讓模型快速記憶片段分布，overfitting 更早發生（ep2 vs ep3）。
- **資料多樣性 > 資料數量**：augmented 3162 筆（含重複）不如 1054 筆真實多樣樣本。
- **擴增到 4125 筆真實樣本後**，val_loss 不再單調發散，證實「真實多樣性」才是有效的正則化。

**探索性實驗（不可直接與上表比較）**

| ID | 評估集 | 樣本 | 問題 |
|----|--------|------|------|
| exp-001 | 訓練集本身（93） | 93 | in-sample |
| exp-002 | 訓練集本身（93） | 93 | in-sample |
| exp-003 | 額外 500 筆（與訓練集高度重疊） | 500 | 部分 in-sample |
| exp-004 | 所有 1003 筆（訓練集 = 評估集） | 1000 | 完全 in-sample，+55.65% 為膨脹數字 |

---

## 跨模型 Zero-Shot 對照（Gemma 4）

在**同一份** held-out test split（520 筆）上，評估 Google Gemma 4 兩種音頻架構的 zero-shot 表現
（`gemma/evaluate_gemma.py`，詳見 [`gemma/README.md`](gemma/README.md) 與 `report/gemma_baseline_results.md`）。

| Exp | 模型 | 音頻架構 | 微調 |
|-----|------|----------|------|
| exp-007 | `google/gemma-4-E4B-it` | 音頻 encoder（USM-style） | 無（zero-shot） |
| exp-008 | `google/gemma-4-12B-it` | encoder-free（raw waveform → linear projection） | 無（zero-shot） |

| 指標 | Qwen Baseline | **Qwen FT（本專案）** | Gemma 12B | Gemma E4B |
|------|---------------|----------------------|-----------|-----------|
| ROUGE-1 | 0.4826 ± 0.1987 | **0.5035 ± 0.2080** | 0.1670 ± 0.0494 | 0.1681 ± 0.0511 |
| ROUGE-2 | 0.2437 ± 0.2173 | **0.2711 ± 0.2355** | 0.0170 ± 0.0144 | 0.0162 ± 0.0141 |
| ROUGE-L | 0.3882 ± 0.2117 | **0.4095 ± 0.2228** | 0.1099 ± 0.0267 | 0.1086 ± 0.0276 |
| BLEU | 0.1798 ± 0.2023 | **0.2044 ± 0.2223** | 0.0005 ± 0.0032 | 0.0009 ± 0.0041 |
| METEOR | 0.4001 ± 0.2148 | **0.4184 ± 0.2297** | 0.1839 ± 0.0346 | 0.1974 ± 0.0365 |

**解讀（勿直接當成「Gemma 比較差」）**

- Gemma 的低分主要來自**輸出風格與長度不匹配**：平均 161.7 詞（12B）/ 181.9 詞（E4B）的 markdown 條列，
  對上 48.4 詞的 reference 散文，precision-based 指標被嚴重懲罰。
- **Template collapse**：encoder-free 12B 有 **65%（339/520）** 的片段塌陷成同一套「high-energy EDM / Hardstyle」模板；
  有 encoder 的 E4B 只有 **17%（90/520）** → 原生音頻 encoder 確實注入了 content-grounded 訊號。
- E4B 在 METEOR 上顯著優於 12B（+0.0135，約 6 SE），其餘指標在雜訊範圍內。
- 結論：即便是更大、更新的通用模型，zero-shot 仍難以對齊領域任務；**PEFT 仍是必要的**。

> 註：兩個 Gemma baseline 使用 `num_beams=1`（12B dense 在 24 GB GPU 上 beam search 會 OOM），
> 與 Qwen 評估的 beam=4 不同，此差異已記錄於 `experiments/runs.json`。

---

## 可解釋性分析（Audio-TAM）

report 中另外實作了 **Audio-TAM**（把 Token Activation Maps（Li et al., ICCV 2025） 從視覺 patch 延伸到音頻時間軸），
用來檢驗「模型生成 `piano` 這個 token 時，是否真的對齊到音頻中鋼琴出現的時間」。

- 代理驗證（MusicCaps 無秒級標注）：silence padding 測試 6/6 通過；content word 的 peakiness 為 function word 的 **1.63 倍**（弱但存在的語意選擇性）。
- BabySlakh 乾淨多軌量化驗證：LM-Head selectivity 僅 **0.087**、Gradient **0.059**、Attention **0.0**（所有 token 塌陷到同一時間位置）。
- 診斷：Qwen2-Audio 屬 decoder-only「audio-as-tokens」架構，**缺少 decoder↔encoder cross-attention**，
  時間對應在 self-attention 全域混合後即遺失 → 這是架構層級限制，並非微調不足。

> Audio-TAM 的分析與圖表在 `report/`（`main.tex` §Audio-TAM、`TAM_result_cropped.png`）；
> **實驗腳本尚未併入本 repo**。

---

## 環境建置

### 步驟 1：安裝套件

```bash
python -m venv .venv
source .venv/bin/activate

# Qwen2-Audio 訓練 / 評估（主要 pipeline）
pip install -r qwen/requirements.txt

# 若要跑 Gemma 4 zero-shot baseline（需 transformers>=5.5.0）
pip install -r gemma/requirements.txt
```

| requirements | 用途 | 主要套件 |
|--------------|------|----------|
| `qwen/requirements.txt` | 訓練 + 評估主 pipeline | torch≥2.2, transformers≥4.45, peft≥0.12, tensorboard, librosa |
| `gemma/requirements.txt` | Gemma 4 zero-shot baseline | transformers≥5.5.0, torchvision, librosa, evaluate |
| `salmonn/requirements.txt` | SALMONN 備查環境（pinned 完整鎖定版） | 全套 pinned 版本 |

### 步驟 2：取得音頻（二擇一）

**選項 A — 使用共享音頻目錄（推薦，組織內部成員）**

音頻已由管理員統一下載並放置於共享路徑。建立 symlink 指向該目錄即可：

```bash
ln -s /path/to/shared/musiccaps_audio ./musiccaps_audio
```

確認可用：

```bash
ls musiccaps_audio/ | wc -l   # 應與 musiccaps_processed.json 筆數（5165）相符
```

**選項 B — 自行從 YouTube 下載（首次建置或新環境）**

```bash
python salmonn/download_musiccaps.py --target all --workers 4
# 受 YouTube 版權與下載成功率限制，目前累積可下載約 5165 筆
```

### 步驟 3：確認切分檔案

切分 JSON 已 commit 至 repo，**clone 後直接可用，不需重跑 split.py**：

```
data/musiccaps_train.json  — 4125 筆（訓練）
data/musiccaps_val.json    —  520 筆（驗證）
data/musiccaps_test.json   —  520 筆（最終評估，訓練期間禁止使用）
```

---

**硬體需求**

| 項目 | 最低要求 |
|------|----------|
| GPU VRAM | 24 GB（已在 RTX 3090 / 4090 測試） |
| CUDA | 11.8+ |
| Python | 3.10+ |
| 磁碟空間 | ~20 GB 以上（原始音頻 ~3.7 GB + 單一 checkpoint 約 ~15 GB。**注意：** 若 `save_total_limit = None` 且保留所有 epoch checkpoints，15 個 epochs 累計會佔用超過 ~225 GB。若硬碟空間有限，建議設定 `save_total_limit` 限制數量，或訓練完成後僅保留輕量化的 `checkpoint-best` adapter） |

> 參考耗時：1267 筆時代的 pilot run 約 2.5 小時；全量 4125 筆 × 15 epochs 約 **15 小時**（單張 RTX 3090）。

---

## 執行流程

### 訓練

```bash
# 在 qwen/train_lora_qwen.py 頂部設定 EXP_ID = "exp-009"（每次實驗前更換；exp-008 已被使用）

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
    --data_json data/musiccaps_val.json \
    --output outputs/eval_baseline_val.json

# 2. 評估微調 checkpoint（從 best checkpoint 讀取）
python qwen/evaluate_qwen.py \
    --lora_path outputs/qwen_musiccaps_finetuned/checkpoint-best \
    --data_json data/musiccaps_val.json \
    --output outputs/eval_exp009.json

# 3. 計算指標並比較（預設自動啟用 1000 輪 Bootstrap 自助法估算均值與標準差）
python run_eval_compare.py \
    --baseline outputs/eval_baseline_val.json \
    --finetuned outputs/eval_exp009.json

# 4. 若要跑傳統 Corpus 層級評估（無標準差），請加上 --no-bootstrap
python run_eval_compare.py \
    --baseline outputs/eval_baseline_val.json \
    --finetuned outputs/eval_exp009.json \
    --no-bootstrap

# 5. 只要 per-clip Mean ± Sample SD（Gemma baseline 採用的計分方式）
python compute_metrics_with_std.py \
    --baseline outputs/eval_baseline_val.json \
    --finetuned outputs/eval_exp009.json

# 6. 查看所有實驗摘要
python experiments/compare.py
python experiments/compare.py --detail exp-008
```

> **評估指標說明**：
> - 預設運行的比較腳本會以 `1000 輪 Bootstrap 自助法` 估算平均分數與其統計標準差。
> - 輸出表格中各欄位的代表意義：
>   - **`Mean ± Boot SD`**：自助法重採樣後的指標平均值，以及其平均值的標準誤差（SEM，寫論文/報告評估統計顯著性使用）。
>   - **`Sample SD`**（資料之間的標準差）：測試集中各個音訊樣本指標分數的標準差（反映模型在不同音樂片段間的表現波動/穩定度，即原本 `compute_metrics_with_std.py` 算出的波動度）。
> - **Test set 只在最終報告時執行一次**，將 `--data_json` 換成 `data/musiccaps_test.json`。

### Gemma 4 Zero-Shot Baseline

```bash
# encoder-based E4B（exp-007）
python gemma/evaluate_gemma.py \
    --model_id google/gemma-4-E4B-it \
    --data_json data/musiccaps_test.json \
    --num_beams 1 \
    --output outputs/eval_gemma4_baseline_test.json

# encoder-free 12B（exp-008；24 GB GPU 上 beam search 會 OOM，必須 num_beams=1）
python gemma/evaluate_gemma.py \
    --model_id google/gemma-4-12B-it \
    --data_json data/musiccaps_test.json \
    --num_beams 1 \
    --output outputs/eval_gemma4_12b_baseline_test.json
```

完整操作步驟（RunPod、模型下載、已知坑）見 [`gemma/README.md`](gemma/README.md)。

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

### 7. 訓練 loss 週期性突刺（全量訓練）

全量 4125 筆訓練在 ep7 / ep12 出現訓練 loss 突刺（6.17 → 9.04、7.52 → 9.29），隨後自行恢復。
推測為 effective batch size = 8 下的梯度變異 / 資料排序效應。
→ 目前不影響 val_loss 趨勢（最佳點仍在 ep3），若要消除可考慮加大 accumulation 或固定 collation 順序。

### 8. Zero-shot 模型的風格落差會壓垮 n-gram 指標

Gemma 4 zero-shot 的 BLEU 近乎 0，主因是輸出 160–180 詞的 markdown 條列，而 reference 是 48 詞散文。
→ **不要只看 BLEU/ROUGE 下結論**，需併看 METEOR 與 template collapse 率等質化分析。

---

## 論文 / 報告

`report/` 內為 IEEE 格式的完整技術報告（中英混排，需 XeLaTeX）。

```bash
cd report
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

| 檔案 | 說明 |
|------|------|
| `report/main.tex` | 報告原始檔（含 Motivation / Related Work / TAM / Results / Summary） |
| `report/main.pdf` | 已編譯的 PDF |
| `report/references.bib` | 參考文獻 |
| `report/plot_loss.py` | 產生 loss 曲線圖（Experiment 3 全量訓練） |
| `report/loss_curve.png` | Experiment 3 loss 曲線 |
| `report/TAM_result_cropped.png` | Audio-TAM 視覺化結果 |
| `report/gemma_baseline_results.md` | Gemma 4 zero-shot 詳細分析 |

> 注意：`main.tex` 另外引用了 `loss_curve_pilot.pdf` / `loss_curve_augmented.pdf` / `loss_curve.pdf`，
> 這些向量檔未 commit 至 repo（`.gitignore` 排除 LaTeX 產物），編譯前需先由 `plot_loss.py` 產生。

---

## 專案結構

```
mllm-music-explanation/
├── data/
│   ├── split.py                       # ← 標準切分腳本（80/10/10，seed=42，aspect-stratified）
│   ├── musiccaps_train.json           # ← 訓練集 4125 筆（標準，由 split.py 生成）
│   ├── musiccaps_val.json             # ← 驗證集 520 筆（標準，由 split.py 生成）
│   ├── musiccaps_test.json            # ← 測試集 520 筆（最終評估才用）
│   ├── create_test_set.py             # 依索引區間切小型子集（smoke test 用）
│   ├── musiccaps_test_1001_1100.json  # 100 筆 smoke-test 子集
│   └── test.json                      # 早期臨時測試集（465 筆，非標準）
│
├── qwen/                              # Qwen2-Audio 微調 pipeline（主要研究）
│   ├── train_lora_qwen.py             # 訓練腳本（修改 EXP_ID 後執行）
│   ├── evaluate_qwen.py               # 推論 + 輸出 prediction JSON
│   ├── augment_dataset.py             # Pitch shift 資料擴增（供參考，exp-006 已證實無效）
│   ├── launch_tb.sh                   # TensorBoard 啟動腳本
│   └── requirements.txt
│
├── gemma/                             # Gemma 4 zero-shot 跨模型對照（exp-007 / exp-008）
│   ├── evaluate_gemma.py              # E4B 與 12B 共用，只換 --model_id
│   ├── README.md                      # 操作指南（RunPod、下載、已知坑）
│   └── requirements.txt
│
├── salmonn/                           # SALMONN baseline（保留備查）
│   ├── train_lora.py
│   ├── evaluate_salmonn.py
│   ├── download_musiccaps.py          # MusicCaps 音頻下載器
│   └── requirements.txt
│
├── experiments/
│   ├── runs.json                      # 所有實驗的完整紀錄（必須維護）
│   └── compare.py                     # CLI 比較表格
│
├── report/                            # IEEE 格式技術報告（XeLaTeX）
│   ├── main.tex / main.pdf / references.bib
│   ├── plot_loss.py / loss_curve.png
│   ├── TAM_result_cropped.png
│   └── gemma_baseline_results.md
│
├── musiccaps_processed.json           # 主資料清單（5165 筆：音頻路徑 + 標注）
├── musiccaps_audio/                   # 已下載的 .wav 音頻（git ignored）
├── run_eval_compare.py                # 評估比較腳本（Corpus 評估 + Bootstrap，含 Sample SD 與 Boot SD）
├── compute_metrics_with_std.py        # per-clip Mean ± Sample SD 計分（Gemma baseline 採用）
└── outputs/                           # checkpoint + eval 結果（git ignored）
```

---

## 參考資料

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) — Alibaba DAMO Academy
- [Gemma 4](https://huggingface.co/google) — Google DeepMind（zero-shot 對照組）
- [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) — Google DeepMind
- [SALMONN](https://github.com/bytedance/SALMONN) — ByteDance Research
- [PEFT / LoRA](https://github.com/huggingface/peft) — Hugging Face
- Token Activation Map to Visually Explain Multimodal LLMs — Li et al., ICCV 2025

---

## License

本專案採用 [MIT License](LICENSE)。
