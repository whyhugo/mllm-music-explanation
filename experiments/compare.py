#!/usr/bin/env python3
"""
Compare experiment runs from experiments/runs.json.

Usage
-----
# Show all completed runs
python experiments/compare.py

# Show specific runs
python experiments/compare.py exp-001 exp-002

# Show a single run in detail
python experiments/compare.py --detail exp-001
"""

import json
import sys
import os
from pathlib import Path

RUNS_FILE = Path(__file__).parent / "runs.json"
METRICS   = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR"]


def load_runs():
    with open(RUNS_FILE) as f:
        return json.load(f)


def fmt(v, width=8):
    if v is None:
        return " " * width
    if isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    return str(v).rjust(width)


def print_metrics_table(runs):
    completed = [r for r in runs if r.get("evaluation")]
    if not completed:
        print("No completed runs with evaluation data.")
        return

    # Header
    col_w = 12
    id_w  = 10
    header = f"{'ID':<{id_w}} {'Name':<28}"
    for m in METRICS:
        header += f"  {m:>{col_w}}"
    header += f"  {'Avg Δ%':>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Baseline row — prefer the most recent / largest eval set
    base_run   = max(completed, key=lambda r: r["evaluation"].get("eval_set_n", 0))
    base       = base_run["evaluation"]["baseline"]
    base_n     = base_run["evaluation"].get("eval_set_n", "?")
    row = f"{'baseline':<{id_w}} {f'Qwen2-Audio-7B base (n={base_n})':<28}"
    for m in METRICS:
        row += f"  {fmt(base.get(m), col_w)}"
    row += f"  {'—':>{col_w}}"
    print(row)
    print("-" * len(header))

    # Each experiment (completed or eval-pending)
    all_runs_to_show = [r for r in runs if r.get("status") not in ("invalid",)]
    for run in all_runs_to_show:
        ev = run.get("evaluation")
        if ev:
            ft  = ev.get("finetuned", {})
            ip  = ev.get("improvement_pct", {})
            n   = ev.get("eval_set_n", "?")
            avg_imp = sum(ip.values()) / len(ip) if ip else 0
            avg_str = f"+{avg_imp:.1f}% (n={n})"
        else:
            ft, ip, avg_str = {}, {}, "eval pending"

        row = f"{run['id']:<{id_w}} {run['name'][:28]:<28}"
        for m in METRICS:
            v = ft.get(m)
            row += f"  {fmt(v, col_w)}"
        row += f"  {avg_str:>{col_w}}"
        print(row)

    print("=" * len(header))


def print_detail(run):
    print(f"\n{'='*60}")
    print(f"  {run['id']} — {run['name']}")
    print(f"  Date   : {run['date']}")
    print(f"  Status : {run['status']}")
    print(f"  Desc   : {run['description']}")
    print(f"{'='*60}")

    # Training config
    t = run.get("training", {})
    print("\n[Training Config]")
    print(f"  epochs       : {t.get('epochs')}")
    print(f"  LR           : {t.get('learning_rate')}")
    print(f"  batch (eff)  : {t.get('per_device_batch_size')} × {t.get('gradient_accumulation_steps')} = {t.get('effective_batch_size')}")
    print(f"  warmup_ratio : {t.get('warmup_ratio')}")
    print(f"  max_grad_norm: {t.get('max_grad_norm')}")
    print(f"  total_steps  : {t.get('total_steps')}")
    print(f"  runtime      : {t.get('runtime_minutes')} min")

    # Model config
    m = run.get("model", {})
    lo = m.get("lora", {})
    print("\n[Model / LoRA]")
    print(f"  base         : {m.get('base')}")
    print(f"  LoRA r       : {lo.get('r')}  alpha={lo.get('alpha')}")
    print(f"  targets      : {lo.get('target_modules')}")
    print(f"  projector    : {'unfrozen' if m.get('projector_unfrozen') else 'frozen'}")
    if m.get("trainable_params"):
        print(f"  trainable    : {m['trainable_params']:,} / {m['total_params']:,} ({m['trainable_pct']:.2f}%)")

    # Loss curve
    lc = run.get("loss_curve", [])
    if lc:
        print("\n[Loss Curve]")
        print(f"  {'step':>6}  {'epoch':>6}  {'loss':>8}  {'grad_norm':>10}  {'lr':>12}")
        for pt in lc:
            print(f"  {pt['step']:>6}  {pt['epoch']:>6.2f}  {pt['loss']:>8.3f}  {pt['grad_norm']:>10.3f}  {pt['lr']:>12.3e}")

    # Evaluation
    ev = run.get("evaluation")
    if ev:
        print("\n[Evaluation]")
        ft = ev.get("finetuned", {})
        ip = ev.get("improvement_pct", {})
        base_ev = ev.get("baseline", {})
        print(f"  {'Metric':<12}  {'Baseline':>10}  {'Fine-tuned':>10}  {'Δ%':>8}")
        print(f"  {'-'*46}")
        for mm in METRICS:
            b = base_ev.get(mm, 0)
            f = ft.get(mm, 0)
            d = ip.get(mm, 0)
            print(f"  {mm:<12}  {b:>10.4f}  {f:>10.4f}  {d:>+8.2f}%")

    # Observations
    obs = run.get("observations", [])
    if obs:
        print("\n[Observations]")
        for o in obs:
            print(f"  • {o}")

    # Issues
    issues = run.get("issues", [])
    if issues:
        print("\n[Issues / TODO for next run]")
        for i in issues:
            print(f"  ✗ {i}")
    print()


def main():
    runs = load_runs()
    args = sys.argv[1:]

    if "--detail" in args:
        args.remove("--detail")
        target_id = args[0] if args else None
        for run in runs:
            if target_id is None or run["id"] == target_id:
                print_detail(run)
        return

    # Filter by IDs if provided
    if args:
        runs = [r for r in runs if r["id"] in args]

    print_metrics_table(runs)

    # Print latest run's observations as a quick summary
    completed = [r for r in runs if r.get("evaluation")]
    if completed:
        latest = completed[-1]
        print(f"\nLatest: {latest['id']} — {latest['name']}")
        for obs in latest.get("observations", [])[:3]:
            print(f"  • {obs}")
        if latest.get("next_experiment"):
            print(f"\n→ Next: {latest['next_experiment']}")


if __name__ == "__main__":
    main()
