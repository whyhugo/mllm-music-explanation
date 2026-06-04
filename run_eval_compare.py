# -*- coding: utf-8 -*-
"""
評估指標比較與多輪彙整腳本 (run_eval_compare.py)
==============================================
這個腳本用來讀取模型推論產生的預測結果 JSON 檔案，計算多項自然語言處理 (NLP) 指標，
並輸出排版美觀的對照表格，同時自動匯出成 CSV 檔案。

主要功能：
  1. 支援載入多輪 (最多 3 輪) 重複實驗的預測 JSON 檔案。
  2. 單輪內統計：計算該輪內「所有測試音樂樣本」得分的平均值 (Mean) 與標準差 (Sample Std)。
  3. 多輪彙整統計：計算 3 輪實驗「平均分數」之間的平均值與標準差 (Run Std)。
  4. 自動將每輪結果與多輪彙整結果存成 CSV 格式，方便用 Excel 或 Pandas 載入分析。

輸出的檔案說明：
  1. 每輪獨立評估的 CSV 結果檔：
     - 檔案路徑：`outputs/eval_run1_results.csv`、`outputs/eval_run2_results.csv`、`outputs/eval_run3_results.csv`
     - 內容：單獨記錄第 1, 2, 3 輪實驗的詳細評估數據。
     - 欄位說明：
       * Metric: 評估指標名稱（如 rouge1、bleu、Average 等）。
       * Baseline Mean: 基準模型在該輪中，所有樣本得分的平均值。
       * Baseline Sample Std: 基準模型在該輪中，所有樣本得分的標準差（代表音樂樣本間的難易度波動，值越小代表對各音樂類型越穩定）。
       * Ours Mean: 我們的微調模型在該輪中，所有樣本得分的平均值。
       * Ours Sample Std: 我們的微調模型在該輪中，所有樣本得分的標準差。
       * Abs Improve: Ours 平均分減去 Baseline 平均分的絕對提升分數 (Ours Mean - Baseline Mean)。
       * Rel Improve (%): Ours 相較於 Baseline 的相對提升百分比。
       
  2. 多輪彙整輪次級穩定度的 CSV 彙整檔：
     - 檔案路徑：`outputs/eval_summary_run_stability.csv`
     - 內容：彙整全部 3 輪實驗結果，計算跨輪的穩定性指標。
     - 欄位說明：
       * Metric: 評估指標名稱（如 rouge1、bleu、Average 等）。
       * Baseline Mean: 基準模型在所有運行的輪次中，各輪平均得分的「總平均值」。
       * Baseline Run Std: 基準模型在所有運行的輪次中，各輪平均得分之間的「標準差」（代表隨機訓練或隨機種子帶來的波動性，值越小代表多次訓練/推論越穩定）。
       * Ours Mean: 我們的微調模型在所有運行的輪次中，各輪平均得分的「總平均值」。
       * Ours Run Std: 我們的微調模型在所有運行的輪次中，各輪平均得分之間的「標準差」。
       * Abs Improve: Ours Mean 與 Baseline Mean 的絕對差值。
       * Rel Improve (%): 相對提升百分比。
"""

import os
import json
import evaluate
import numpy as np
import pandas as pd

# 嘗試載入 NLTK 庫來計算平滑化後的 Sentence-level BLEU 分數。
# Sentence-level BLEU 若不使用平滑化 (Smoothing)，常因單句 4-gram 匹配不到而直接得 0 分，導致變異數極不穩定。
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

def load_data(file_path):
    """
    從指定的 JSON 檔案載入推論預測資料。
    
    參數:
        file_path (str): 預測 JSON 檔案的路徑。
    
    回傳:
        preds (list): 模型生成的文字描述列表。
        refs (list): 資料集中的真實文字描述 (Ground Truth) 列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取模型生成的文字與對應的真實標籤文字
    preds = [d['generated_text'] for d in data]
    refs = [d['ground_truth_caption'] for d in data]
    return preds, refs

def evaluate_metrics(preds, refs):
    """
    計算預測描述與真實描述之間的各項 NLP 指標。
    會針對單一輪內的所有資料筆數，逐筆計算出各樣本的獨立得分，以求得樣本間的標準差。
    
    參數:
        preds (list): 預測文字列表。
        refs (list): 真實標籤文字列表。
        
    回傳:
        res (dict): 包含各指標 {'指標名稱': (樣本平均值, 樣本標準差)} 的字典。
    """
    # 載入 Hugging Face evaluate 套件中的評估模組
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    
    # 【ROUGE 計算】：
    # 帶入 use_aggregator=False，這樣 ROUGE 就會回傳每一筆樣本的獨立分數列表，而不是只給平均分。
    rouge_res = rouge.compute(predictions=preds, references=refs, use_aggregator=False)
    rouge1_list = rouge_res['rouge1']
    rouge2_list = rouge_res['rouge2']
    rougeL_list = rouge_res['rougeL']
    
    # 用於儲存每一筆樣本獨立 BLEU 與 METEOR 分數的列表
    bleu_list = []
    meteor_list = []
    
    # 如果 NLTK 可用，則初始化平滑函數，避免單句匹配度低時 BLEU score 直接歸零
    smooth = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    # 逐筆計算單一單句的評估指標 (Sentence-level score)
    for p, r in zip(preds, refs):
        # ── 1. 計算單句 BLEU 分數 ──
        if NLTK_AVAILABLE:
            # NLTK sentence_bleu 接收 token list 格式，所以需要對句子做 split() 切詞
            b = sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
        else:
            # 若 NLTK 不可用，則 fallback 慢速使用 evaluate 套件計算單句 BLEU
            try:
                b_res = bleu.compute(predictions=[p], references=[[r]])
                b = b_res['bleu']
            except Exception:
                b = 0.0
        bleu_list.append(b)
        
        # ── 2. 計算單句 METEOR 分數 ──
        try:
            m_res = meteor.compute(predictions=[p], references=[r])
            m = m_res['meteor']
        except Exception:
            m = 0.0
        meteor_list.append(m)
        
    # 計算這批樣本的平均值與樣本標準差 (Sample Standard Deviation)
    res = {}
    for name, lst in [('rouge1', rouge1_list), ('rouge2', rouge2_list), ('rougeL', rougeL_list), ('bleu', bleu_list), ('meteor', meteor_list)]:
        mean_val = np.mean(lst) if lst else 0.0
        # ddof=1 代表計算「樣本標準差」（分母除以 N-1），若只有 1 筆資料則給 0.0 防止出錯
        std_val = np.std(lst, ddof=1) if len(lst) > 1 else 0.0
        res[name] = (mean_val, std_val)
        
    return res

def main():
    # 定義要計算的 5 個核心 NLP 指標
    metrics_list = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']
    
    # 用來收集每一輪所有指標的評估結果，之後用於多輪彙整計算
    base_runs_data = []
    ft_runs_data = []
    
    print("Evaluating runs...")
    # 逐一檢查與處理 3 輪實驗結果
    for i in range(1, 4):
        # 決定這一輪 Baseline JSON 的讀取路徑
        base_path = f'outputs/eval_baseline_run{i}.json'
        # 貼心防呆：如果是第一輪且找不到後綴 _run1 的檔案，則自動回退讀取沒有後綴的預設檔名
        if not os.path.exists(base_path) and i == 1:
            base_path = 'outputs/eval_baseline.json'
            
        # 決定這一輪 Ours (Fine-tuned) JSON 的讀取路徑
        ft_path = f'outputs/eval_finetuned_run{i}.json'
        # 同樣防呆：第一輪找不到後綴 _run1 時，自動回退讀取無後綴的預設檔名
        if not os.path.exists(ft_path) and i == 1:
            ft_path = 'outputs/eval_finetuned.json'
            
        # 只有當這一輪的 Baseline 與 Ours 檔案同時存在時，才執行該輪的評估與輸出
        if os.path.exists(base_path) and os.path.exists(ft_path):
            print(f"\nProcessing Run {i}...")
            print(f"  Baseline: {base_path}")
            print(f"  Ours    : {ft_path}")
            
            # ── 1. 載入並計算該輪的 Baseline 分數 ──
            base_preds, base_refs = load_data(base_path)
            base_res = evaluate_metrics(base_preds, base_refs)
            base_runs_data.append(base_res)
            
            # ── 2. 載入並計算該輪的 Ours 分數 ──
            ft_preds, ft_refs = load_data(ft_path)
            ft_res = evaluate_metrics(ft_preds, ft_refs)
            ft_runs_data.append(ft_res)
            
            # ── 3. 建立並輸出該輪的詳細對照表 ──
            run_rows = []
            for m in metrics_list:
                b_m, b_s = base_res[m]  # 取得 Baseline 該指標的 (平均值, 樣本標準差)
                f_m, f_s = ft_res[m]    # 取得 Ours 該指標的 (平均值, 樣本標準差)
                
                # 計算 Ours 相對 Baseline 的絕對提升分數與提升百分比
                abs_imp = f_m - b_m
                rel_imp = (abs_imp / b_m * 100) if b_m > 0 else 0.0
                
                # 收集這一列的資料
                run_rows.append({
                    'Metric': m,
                    'Baseline Mean': b_m,
                    'Baseline Sample Std': b_s,
                    'Ours Mean': f_m,
                    'Ours Sample Std': f_s,
                    'Abs Improve': abs_imp,
                    'Rel Improve (%)': rel_imp
                })
            
            # 將資料轉換成 Pandas DataFrame 格式，方便整理與輸出
            run_df = pd.DataFrame(run_rows)
            
            # 計算該輪所有指標的平均值，並作為最後一行 (Average row) 串接到表格底部
            avg_row = pd.DataFrame([{
                'Metric': 'Average',
                'Baseline Mean': run_df['Baseline Mean'].mean(),
                'Baseline Sample Std': run_df['Baseline Sample Std'].mean(),
                'Ours Mean': run_df['Ours Mean'].mean(),
                'Ours Sample Std': run_df['Ours Sample Std'].mean(),
                'Abs Improve': run_df['Abs Improve'].mean(),
                'Rel Improve (%)': run_df['Rel Improve (%)'].mean()
            }])
            run_df = pd.concat([run_df, avg_row], ignore_index=True)
            
            # 將結果儲存成 CSV 檔案 (包含完整未格式化的浮點數，便於後續運算)
            os.makedirs('outputs', exist_ok=True)
            run_csv_path = f'outputs/eval_run{i}_results.csv'
            run_df.to_csv(run_csv_path, index=False)
            
            # ── 4. 美化終端機輸出表格的字串排版 ──
            formatted_df = pd.DataFrame()
            formatted_df['Metric'] = run_df['Metric']
            formatted_df['Baseline Mean'] = run_df['Baseline Mean'].map(lambda x: f"{x:.4f}")
            formatted_df['Baseline Sample Std'] = run_df['Baseline Sample Std'].map(lambda x: f"{x:.4f}")
            formatted_df['Ours Mean'] = run_df['Ours Mean'].map(lambda x: f"{x:.4f}")
            formatted_df['Ours Sample Std'] = run_df['Ours Sample Std'].map(lambda x: f"{x:.4f}")
            formatted_df['Abs Improve'] = run_df['Abs Improve'].map(lambda x: f"{x:+.4f}")
            formatted_df['Rel Improve (%)'] = run_df['Rel Improve (%)'].map(lambda x: f"{x:+.2f}%")
            
            # 輸出美化後的單輪表格至終端機
            print(f"\n=== Run {i} 評估結果 (單輪內樣本級 Mean & Std Dev) ===")
            print(formatted_df.to_string(index=False))
            print(f"Saved Run {i} results to {run_csv_path}")
        else:
            # 檔案缺漏時印出警告
            if not os.path.exists(base_path):
                print(f"  Run {i} skipped: {base_path} not found.")
            elif not os.path.exists(ft_path):
                print(f"  Run {i} skipped: {ft_path} not found.")
                
    # 檢查是否有任何一輪成功運行，若無則中止
    if len(base_runs_data) == 0 or len(ft_runs_data) == 0:
        print("\n沒有足夠的評估數據來彙整。")
        return
        
    num_runs = len(base_runs_data)

    # ==================================================================
    # 彙整輸出：計算 3 輪實驗「平均分數」之間的平均值與「跨輪標準差」 (Run Std)
    # ==================================================================
    run_rows = []
    for m in metrics_list:
        # 提取每一輪中，該指標的平均分數 (單輪 Mean)
        b_means = [run[m][0] for run in base_runs_data]
        f_means = [run[m][0] for run in ft_runs_data]
        
        # 計算多輪平均分的「平均值 (Average of Means)」
        b_mean = np.mean(b_means)
        f_mean = np.mean(f_means)
        
        # 計算多輪平均分之間的「跨輪標準差 (Run Standard Deviation)」
        # 說明：這個標準差代表了多次實驗（例如隨機權重初始化或隨機推論）之間的穩定度，而非單一輪內樣本間的難易度變異。
        b_run_std = np.std(b_means, ddof=1) if len(b_means) > 1 else 0.0
        f_run_std = np.std(f_means, ddof=1) if len(f_means) > 1 else 0.0
        
        # 計算彙整後的絕對提升分數與提升百分比
        abs_imp = f_mean - b_mean
        rel_imp = (abs_imp / b_mean * 100) if b_mean > 0 else 0.0
        
        run_rows.append({
            'Metric': m,
            'Baseline Mean': b_mean,
            'Baseline Run Std': b_run_std,
            'Ours Mean': f_mean,
            'Ours Run Std': f_run_std,
            'Abs Improve': abs_imp,
            'Rel Improve (%)': rel_imp
        })
    
    # 建立多輪彙整 DataFrame
    run_df = pd.DataFrame(run_rows)
    
    # 計算多輪彙整各欄位的平均，串接到表格底部作為最後一列 (Average row)
    avg_row_r = pd.DataFrame([{
        'Metric': 'Average',
        'Baseline Mean': run_df['Baseline Mean'].mean(),
        'Baseline Run Std': run_df['Baseline Run Std'].mean(),
        'Ours Mean': run_df['Ours Mean'].mean(),
        'Ours Run Std': run_df['Ours Run Std'].mean(),
        'Abs Improve': run_df['Abs Improve'].mean(),
        'Rel Improve (%)': run_df['Rel Improve (%)'].mean()
    }])
    run_df = pd.concat([run_df, avg_row_r], ignore_index=True)
    
    # 匯出多輪彙整的 CSV 檔案
    summary_csv_path = 'outputs/eval_summary_run_stability.csv'
    run_df.to_csv(summary_csv_path, index=False)
    
    # 美化跨輪彙整表格字串排版
    fmt_run_df = pd.DataFrame()
    fmt_run_df['Metric'] = run_df['Metric']
    fmt_run_df['Baseline Mean'] = run_df['Baseline Mean'].map(lambda x: f"{x:.4f}")
    fmt_run_df['Baseline Run Std'] = run_df['Baseline Run Std'].map(lambda x: f"{x:.4f}")
    fmt_run_df['Ours Mean'] = run_df['Ours Mean'].map(lambda x: f"{x:.4f}")
    fmt_run_df['Ours Run Std'] = run_df['Ours Run Std'].map(lambda x: f"{x:.4f}")
    fmt_run_df['Abs Improve'] = run_df['Abs Improve'].map(lambda x: f"{x:+.4f}")
    fmt_run_df['Rel Improve (%)'] = run_df['Rel Improve (%)'].map(lambda x: f"{x:+.2f}%")
    
    # 輸出美化後的彙整表格至終端機
    print(f"\n=== 多輪彙整：輪次級穩定度 (Run-level Mean & Run Std over {num_runs} runs) ===")
    print(fmt_run_df.to_string(index=False))
    print(f"Saved aggregated run-level stability summary to {summary_csv_path}")

if __name__ == '__main__':
    main()
