# Gemma 4 Zero-Shot Baseline — Evaluation Results

> Working notes to fold into `report/main.tex` (Experiments / Results section).
> Compiled 2026-06-21. Source predictions: `outputs/eval_gemma4_12b_baseline_test.json`
> (exp-008, encoder-free 12B) and `outputs/eval_gemma4_baseline_test.json`
> (exp-007, encoder-based E4B). Both scored.

## 1. Purpose

Cross-model **zero-shot** (no fine-tuning) reference points for music captioning,
evaluated on the **same canonical held-out MusicCaps test split**
(`data/musiccaps_test.json`, **520 clips**) used by the Qwen2-Audio experiments,
so the model families are directly comparable. The prompt and decoding regime are
kept identical to `qwen/evaluate_qwen.py` for fairness (see caveat on beam width
for the 12B variant).

Prompt used for every clip:
`"Please describe this music in detail and list its aspects."`

## 2. Experimental setup

| Field | exp-008 | exp-007 |
|-------|---------|---------|
| Model | `google/gemma-4-12B-it` | `google/gemma-4-E4B-it` |
| Params | ~11.95 B (dense) | E4B |
| Audio architecture | **Encoder-free** — raw waveform → linear projection → LLM embedding | Native audio **encoder** (USM-style) |
| Fine-tuning | none (zero-shot) | none (zero-shot) |
| Decoding | `num_beams=1`, `max_new_tokens=256`, `do_sample=False`, `repetition_penalty=1.1` | **same** (`num_beams=1`, `max_new_tokens=256`) — aligned to 12B for a clean encoder ablation |
| Test set | 520 clips (held-out) | 520 clips (held-out) |
| Predictions | `outputs/eval_gemma4_12b_baseline_test.json` ✅ | `outputs/eval_gemma4_baseline_test.json` ✅ |
| Status | scored | scored |

> **Decoding note:** both variants use `num_beams=1` so the only changing factor
> is the audio architecture (encoder-free vs encoder). This diverges from the
> Qwen2-Audio baseline's `num_beams=4`; note when comparing against Qwen.
> E4B additionally requires `torch>=2.6` for its encoder attention (12B ran on
> torch 2.4.1) — an environment difference, not an experimental-design one;
> generation is deterministic so this does not confound the outputs.

## 3. Results — encoder-free 12B vs encoder-based E4B

Scored with `compute_metrics_with_std.py` over all 520 items. `Mean ± Std` is the
mean and **population** standard deviation of the **per-clip** score across the
520 test clips (i.e. between-sample spread, not a confidence interval / standard
error). Std is reported to gauge how stable the score is across clips.

| Metric  | 12B encoder-free (exp-008) | E4B encoder (exp-007) | Δ (E4B − 12B) |
|---------|----------------------------|-----------------------|---------------|
| ROUGE-1 | 0.1670 ± 0.0494            | 0.1681 ± 0.0511       | +0.0011       |
| ROUGE-2 | 0.0170 ± 0.0144            | 0.0162 ± 0.0141       | −0.0008       |
| ROUGE-L | 0.1099 ± 0.0267            | 0.1086 ± 0.0276       | −0.0013       |
| BLEU    | 0.0005 ± 0.0032            | 0.0009 ± 0.0041       | +0.0004       |
| METEOR  | 0.1839 ± 0.0346            | 0.1974 ± 0.0365       | **+0.0135**   |

To convert to standard error of the mean: SEM = Std / √520 ≈ Std / 22.8
(e.g. METEOR SEM ≈ 0.0016 each model).

**Reading the deltas:** ROUGE and BLEU differences (≤0.0013) are far smaller than
the per-clip std (~0.03–0.05) and are not meaningful — on surface n-gram overlap
the audio encoder makes essentially no difference. The exception is **METEOR
+0.0135**: with a difference-SE ≈ √(0.0016²+0.0016²) ≈ 0.0023, that is ~6 SE, a
small but statistically real improvement for the encoder-based model. The
qualitative analysis (§4) explains why the headline metrics barely move while
behaviour changes substantially.

### LaTeX snippet (drop into main.tex)

```latex
\begin{table}[t]
\caption{Zero-shot Gemma 4 baselines on the held-out MusicCaps test split
(520 clips), same decoding ($num\_beams{=}1$, $max\_new\_tokens{=}256$).
Values are mean $\pm$ per-clip standard deviation.}
\label{tab:gemma_zeroshot}
\centering
\begin{tabular}{lcc}
\hline
Metric & 12B (encoder-free) & E4B (encoder) \\
\hline
ROUGE-1 & $0.1670 \pm 0.0494$ & $0.1681 \pm 0.0511$ \\
ROUGE-2 & $0.0170 \pm 0.0144$ & $0.0162 \pm 0.0141$ \\
ROUGE-L & $0.1099 \pm 0.0267$ & $0.1086 \pm 0.0276$ \\
BLEU    & $0.0005 \pm 0.0032$ & $0.0009 \pm 0.0041$ \\
METEOR  & $0.1839 \pm 0.0346$ & $\mathbf{0.1974 \pm 0.0365}$ \\
\hline
\end{tabular}
\end{table}
```

## 4. Qualitative observations (important for the write-up)

The n-gram metrics (especially BLEU/ROUGE-2) are very low. The dominant cause is
**not** that the model says nothing useful — it is an **output-style mismatch**
against the short, prose-style MusicCaps references:

- **Verbosity:** mean generated length is **161.7 words** vs **48.4 words** for the
  references — ~3.3× longer. Long outputs dilute n-gram precision (BLEU) heavily.
- **Formatting:** outputs are Markdown — bold headers (`### 1. Genre & Style`),
  bullet lists, etc. — whereas references are a single flowing paragraph. The
  structural tokens never match references and depress ROUGE/BLEU.
- **Verbosity (E4B):** E4B is even longer — mean **181.9 words** vs the 12B's
  161.7 and the references' 48.4. Both Gemma variants share the same verbose,
  Markdown-heavy instruction-tuned style, which is why the surface metrics for the
  two are so close.

### The key finding: the encoder breaks template-collapse but does not fix grounding

Both models were spot-checked on the same 20 evenly-spaced clips, and the
"defaults to a generic EDM/dance label" pattern was counted across all 520.

- **Genre accuracy is poor for both** (~3/20, 15–20% of predicted genres match the
  reference) — neither zero-shot model reliably identifies what it is hearing.
- **But the failure mode differs sharply.** The encoder-free 12B **collapses onto a
  single generic "high-energy EDM / Hardstyle" template for 65% (339/520) of all
  clips**, regardless of the true content (jazz, country, choir, french horn, Irish
  uilleann pipes, orchestral all labelled EDM). The encoder-based E4B does this
  only **17% (90/520)** of the time; its (still mostly-wrong) guesses are *diverse*
  — Afrobeat, Indian classical, Balkan folk, ambient/downtempo, country.

**Interpretation:** the native audio encoder injects a genuinely audio-dependent
signal that prevents the degenerate single-template collapse seen in the
encoder-free model (and yields the small, real METEOR gain). However, that signal
is too weak for accurate zero-shot genre/content identification — E4B produces
*varied* wrong answers rather than *uniform* wrong answers. So the audio encoder
helps, but neither off-the-shelf model achieves reliable grounding without
fine-tuning.

| Behaviour (520 clips) | 12B encoder-free | E4B encoder |
|-----------------------|:----------------:|:-----------:|
| Generic EDM-template rate | **65%** (339/520) | **17%** (90/520) |
| Genre match (20-clip manual) | ~15–20% | ~15–20% |
| Mean generated length (words) | 161.7 | 181.9 |

Takeaway for the report: surface-overlap metrics under-credit fluent but verbose,
differently-formatted zero-shot models and barely separate the two architectures;
the meaningful difference is qualitative (template-collapse vs diverse-but-wrong)
plus a small METEOR gain. This motivates (a) reporting METEOR alongside
ROUGE/BLEU, (b) qualitative grounding analysis, and (c) our fine-tuning, which
should align both output style and audio grounding to the target captions.
