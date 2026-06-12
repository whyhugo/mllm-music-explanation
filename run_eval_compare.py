import json
import argparse
import evaluate
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation metrics between baseline and fine-tuned models")
    parser.add_argument('--baseline', type=str, default='outputs/eval_baseline.json', help='Path to baseline evaluation results')
    parser.add_argument('--finetuned', type=str, default='outputs/eval_ours.json', help='Path to fine-tuned evaluation results')
    args = parser.parse_args()

    print(f"Loading Baseline data from {args.baseline}...")
    base_preds, base_refs = load_data(args.baseline)
    
    print(f"Loading Fine-tuned data from {args.finetuned}...")
    ft_preds, ft_refs = load_data(args.finetuned)
    
    print("\nEvaluating Baseline...")
    base_metrics = evaluate_metrics(base_preds, base_refs)
    
    print("Evaluating Fine-tuned...")
    ft_metrics = evaluate_metrics(ft_preds, ft_refs)
    
    print("\n--- Quantitative Comparison ---")
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
