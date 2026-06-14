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

def evaluate_metrics(preds, refs):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=refs)
    meteor_res = meteor.compute(predictions=preds, references=refs)
    
    return {
        'rouge1': rouge_res['rouge1'],
        'rouge2': rouge_res['rouge2'],
        'rougeL': rouge_res['rougeL'],
        'bleu': bleu_res['bleu'],
        'meteor': meteor_res['meteor']
    }

def get_sentence_level_metrics(preds, refs):
    """Pre-calculate sentence-level metrics to make bootstrap resampling extremely fast."""
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'bleu': [],
        'meteor': []
    }
    
    print("Calculating sentence-level metrics...")
    for p, r in tqdm(zip(preds, refs), total=len(preds)):
        # Compute individually
        r_res = rouge.compute(predictions=[p], references=[r])
        b_res = bleu.compute(predictions=[p], references=[r])
        m_res = meteor.compute(predictions=[p], references=[r])
        
        scores['rouge1'].append(r_res['rouge1'])
        scores['rouge2'].append(r_res['rouge2'])
        scores['rougeL'].append(r_res['rougeL'])
        scores['bleu'].append(b_res['bleu'])
        scores['meteor'].append(m_res['meteor'])
        
    return {k: np.array(v) for k, v in scores.items()}

def run_bootstrap(scores, rounds=1000, seed=42):
    """Run bootstrap resampling on pre-calculated sentence-level scores."""
    np.random.seed(seed)
    N = len(list(scores.values())[0])
    
    bootstrap_means = {k: [] for k in scores.keys()}
    
    for _ in range(rounds):
        indices = np.random.choice(N, size=N, replace=True)
        for k in scores.keys():
            bootstrap_means[k].append(scores[k][indices].mean())
            
    results = {}
    for k in scores.keys():
        results[k] = {
            'mean': np.mean(bootstrap_means[k]),
            'std': np.std(bootstrap_means[k])
        }
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation metrics between baseline and fine-tuned models")
    parser.add_argument('--baseline', type=str, default='outputs/eval_baseline.json', help='Path to baseline evaluation results')
    parser.add_argument('--finetuned', type=str, default='outputs/eval_ours.json', help='Path to fine-tuned evaluation results')
    parser.add_argument('--no-bootstrap', dest='bootstrap', action='store_false', help='Disable bootstrap resampling (run standard corpus-level evaluation instead)')
    parser.add_argument('--rounds', type=int, default=1000, help='Number of bootstrap rounds')
    parser.set_defaults(bootstrap=True)
    args = parser.parse_args()

    print(f"Loading Baseline data from {args.baseline}...")
    base_preds, base_refs = load_data(args.baseline)
    
    print(f"Loading Fine-tuned data from {args.finetuned}...")
    ft_preds, ft_refs = load_data(args.finetuned)
    
    if len(base_preds) != len(ft_preds):
        print(f"Warning: Baseline has {len(base_preds)} samples, but Fine-tuned has {len(ft_preds)} samples.")

    if args.bootstrap:
        print(f"\n--- Running Bootstrap Resampling ({args.rounds} rounds) ---")
        print("Processing Baseline...")
        base_scores = get_sentence_level_metrics(base_preds, base_refs)
        base_bootstrap = run_bootstrap(base_scores, rounds=args.rounds)
        
        print("\nProcessing Fine-tuned...")
        ft_scores = get_sentence_level_metrics(ft_preds, ft_refs)
        ft_bootstrap = run_bootstrap(ft_scores, rounds=args.rounds)
        
        print("\n--- Quantitative Comparison (Bootstrap Resampling) ---")
        metrics_list = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']
        print(f"{'Metric':<10} | {'Baseline (Mean ± Boot SD)':<28} | {'Baseline Sample SD':<18} | {'Fine-tuned (Mean ± Boot SD)':<28} | {'Fine-tuned Sample SD':<20} | {'Improvement':<12}")
        print("-" * 120)
        for m in metrics_list:
            b_mean = base_bootstrap[m]['mean']
            b_sample_std = base_scores[m].std()
            b_bootstrap_std = base_bootstrap[m]['std']
            
            f_mean = ft_bootstrap[m]['mean']
            f_sample_std = ft_scores[m].std()
            f_bootstrap_std = ft_bootstrap[m]['std']
            
            imp = (f_mean - b_mean) / b_mean * 100 if b_mean > 0 else 0
            
            b_mean_str = f"{b_mean:.4f} ± {b_bootstrap_std:.4f}"
            b_sample_std_str = f"{b_sample_std:.4f}"
            f_mean_str = f"{f_mean:.4f} ± {f_bootstrap_std:.4f}"
            f_sample_std_str = f"{f_sample_std:.4f}"
            
            print(f"{m:<10} | {b_mean_str:<28} | {b_sample_std_str:<18} | {f_mean_str:<28} | {f_sample_std_str:<20} | {imp:+.2f}%")
    else:
        print("\nEvaluating Baseline (Corpus-level)...")
        base_metrics = evaluate_metrics(base_preds, base_refs)
        
        print("Evaluating Fine-tuned (Corpus-level)...")
        ft_metrics = evaluate_metrics(ft_preds, ft_refs)
        
        print("\n--- Quantitative Comparison (Corpus-level) ---")
        metrics_list = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']
        print(f"{'Metric':<10} | {'Baseline':<10} | {'Fine-tuned':<10} | {'Improvement':<10}")
        print("-" * 50)
        for m in metrics_list:
            b_val = base_metrics[m]
            f_val = ft_metrics[m]
            imp = (f_val - b_val) / b_val * 100 if b_val > 0 else 0
            print(f"{m:<10} | {b_val:<10.4f} | {f_val:<10.4f} | {imp:+.2f}%")

if __name__ == '__main__':
    main()
