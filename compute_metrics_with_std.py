import json
import argparse
import evaluate
import numpy as np
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    preds = [d['generated_text'] for d in data]
    refs = [d['ground_truth_caption'] for d in data]
    return preds, refs

def get_item_metrics(preds, refs):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    
    results = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'meteor': []}
    
    for p, r in tqdm(zip(preds, refs), total=len(preds)):
        # Compute individually
        r_res = rouge.compute(predictions=[p], references=[r])
        b_res = bleu.compute(predictions=[p], references=[r])
        m_res = meteor.compute(predictions=[p], references=[r])
        
        results['rouge1'].append(r_res['rouge1'])
        results['rouge2'].append(r_res['rouge2'])
        results['rougeL'].append(r_res['rougeL'])
        results['bleu'].append(b_res['bleu'])
        results['meteor'].append(m_res['meteor'])
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Compute metrics with standard deviation for statistical significance")
    parser.add_argument('--baseline', type=str, default='outputs/eval_baseline.json', help='Path to baseline evaluation results')
    parser.add_argument('--finetuned', type=str, default='outputs/eval_ours.json', help='Path to fine-tuned evaluation results')
    args = parser.parse_args()

    print(f"Loading Baseline data from {args.baseline}...")
    base_preds, base_refs = load_data(args.baseline)
    
    print(f"Loading Fine-tuned data from {args.finetuned}...")
    ft_preds, ft_refs = load_data(args.finetuned)
    
    print("\nCalculating sentence-level metrics for Baseline...")
    base_res = get_item_metrics(base_preds, base_refs)
    
    print("Calculating sentence-level metrics for Fine-tuned model...")
    ft_res = get_item_metrics(ft_preds, ft_refs)
    
    print(f"\n--- Metrics with Standard Deviation (Test Set: {len(base_preds)} items) ---")
    print(f"{'Metric':<10} | {'Baseline (Mean ± Std)':<25} | {'Ours (Mean ± Std)':<25}")
    print("-" * 65)
    for m in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']:
        b_mean = np.mean(base_res[m])
        b_std = np.std(base_res[m])
        f_mean = np.mean(ft_res[m])
        f_std = np.std(ft_res[m])
        print(f"{m:<10} | {b_mean:.4f} ± {b_std:.4f}       | {f_mean:.4f} ± {f_std:.4f}")

if __name__ == '__main__':
    main()
