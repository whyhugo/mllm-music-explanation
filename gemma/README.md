# Gemma 4 Audio Baselines — Operations Guide

Zero-shot MusicCaps captioning baselines using Google's **Gemma 4** audio
models, scored against the Qwen2-Audio experiments on the **same held-out test
split** (`data/musiccaps_test.json`, 520 clips).

| Exp | Model | Audio architecture | Predictions file |
|-----|-------|--------------------|------------------|
| exp-007 | `google/gemma-4-E4B-it` | audio **encoder** (USM-style) | `outputs/eval_gemma4_baseline_test.json` |
| exp-008 | `google/gemma-4-12B-it` | **encoder-free** (raw waveform → linear projection → LLM) | `outputs/eval_gemma4_12b_baseline_test.json` |

Both are **zero-shot** (no fine-tuning). One script, [`evaluate_gemma.py`](evaluate_gemma.py),
serves both — the multimodal API is identical, only `--model_id` changes. The
prompt and decoding params match `qwen/evaluate_qwen.py` for a fair comparison.

Output schema is identical to `qwen/evaluate_qwen.py`:
`{wav_path, generated_text, ground_truth_caption, aspect_list}`.

---

## 1. Prerequisites (on the pod, one-time)

```bash
cd /workspace/mllm-music-explanation
git fetch origin && git checkout exp/gemma4-baseline && git pull

# Extra deps beyond the base env (Gemma 4 processor needs these):
.venv/bin/pip install -r gemma/requirements.txt
# Key ones: transformers>=5.5.0, torchvision (processor imports it at load),
#           librosa (audio decoding).
```

Notes:
- `google/gemma-4-*-it` probed as **not gated** → downloads without an HF token.
  If that changes, run `huggingface-cli login` or `export HF_TOKEN=...` first.
- Model downloads land in `/workspace/.cache` (persistent), so they survive a
  pod stop/restart and are only downloaded once.

---

## 2. Connect + keep the run alive (tmux)

The RunPod SSH session is interactive — if it disconnects, a foreground job is
killed. Use **tmux** (or `nohup ... &`) for the multi-hour runs.

```bash
tmux new -s gemma          # start a session
#   ... run an eval command (below) ...
# Ctrl-b then d            # detach (job keeps running)
tmux attach -t gemma       # reattach later to watch progress
```

---

## 3. Run the evaluations

Run from the repo root (`/workspace/mllm-music-explanation`).

### exp-007 — Gemma 4 E4B (encoder-based)
```bash
.venv/bin/python gemma/evaluate_gemma.py \
    --output outputs/eval_gemma4_baseline_test.json
```
Defaults: full 520-sample test set, `num_beams=4`, `max_new_tokens=256`.
Throughput ≈ 20 s/sample → ~3 h for the full set.

### exp-008 — Gemma 4 12B (encoder-free, raw waveform)
```bash
.venv/bin/python gemma/evaluate_gemma.py \
    --model_id google/gemma-4-12B-it \
    --num_beams 1 \
    --output outputs/eval_gemma4_12b_baseline_test.json
```
**Why `--num_beams 1`:** the dense 12B is ~24 GB in bf16 — on a 24 GB 3090,
beam search (beam=4) KV-cache will very likely OOM. beam=1 is the safe choice
(it diverges slightly from the Qwen beam=4 decoding — note the value used in
`experiments/runs.json` exp-008). If you have a bigger GPU, raise `--num_beams 4`
to match exactly.

> ⚠️ **One model at a time.** A 24 GB GPU cannot hold E4B and 12B together —
> finish/stop one before starting the other.

Quick smoke test before a full run (any model):
```bash
.venv/bin/python gemma/evaluate_gemma.py --limit 3 --output /tmp/smoke.json --model_id <id>
```

---

## 4. Auto-stop the pod when the run finishes (save GPU cost)

`runpodctl` is pre-installed and `RUNPOD_POD_ID` is set, but you must configure
the API key **once**:

```bash
runpodctl config --apiKey <YOUR_RUNPOD_API_KEY>   # one-time
```

Then chain a stop after the run. Use `;` (not `&&`) so the pod stops **even if
the eval crashes** — otherwise a failed job leaves the GPU billing overnight:

```bash
.venv/bin/python gemma/evaluate_gemma.py \
    --output outputs/eval_gemma4_baseline_test.json ; \
    runpodctl stop pod $RUNPOD_POD_ID
```

Or eval → metrics → stop in one chain:
```bash
.venv/bin/python gemma/evaluate_gemma.py \
    --output outputs/eval_gemma4_baseline_test.json && \
.venv/bin/python compute_metrics_with_std.py \
    --baseline outputs/eval_baseline_test.json \
    --finetuned outputs/eval_gemma4_baseline_test.json ; \
    runpodctl stop pod $RUNPOD_POD_ID
```

- `stop pod` **preserves** the persistent volume (`/workspace`, outputs, model
  cache). Restart the pod later to continue. **Never** use `remove` — that
  destroys the pod.
- Outputs are written to `/workspace` (persistent), so they survive the stop.

---

## 5. Score the results

`compute_metrics_with_std.py` (repo root) reports ROUGE-1/2/L, BLEU, METEOR with
per-item std, comparing any predictions file against a baseline.

```bash
# Gemma E4B vs Qwen2-Audio zero-shot baseline (same 520-sample test set)
.venv/bin/python compute_metrics_with_std.py \
    --baseline outputs/eval_baseline_test.json \
    --finetuned outputs/eval_gemma4_baseline_test.json

# Gemma 12B
.venv/bin/python compute_metrics_with_std.py \
    --baseline outputs/eval_baseline_test.json \
    --finetuned outputs/eval_gemma4_12b_baseline_test.json
```

After scoring, fill the `evaluation` block and set `status: "completed"` for the
matching entry (exp-007 / exp-008) in `experiments/runs.json`.

---

## 6. Pull results back to the local machine (optional)

From your local machine (direct-TCP SSH supports scp; the proxy connection does
not):

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
    root@<pod-ip>:/workspace/mllm-music-explanation/outputs/eval_gemma4_*_test.json \
    outputs/
```
Get `<pod-ip>:<port>` from the pod's **Connect → SSH over exposed TCP** panel.
