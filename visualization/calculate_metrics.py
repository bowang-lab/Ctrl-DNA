#!/usr/bin/env python
# coding: utf-8

import os
import re
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# import your utility functions
from metrics_utils import (
    process_file,
    compute_tradeoff_score,
    positional_entropy,
    gc_content,
    gc_distance,
    compute_motif_corr
)
import scripts.motifs, scripts.utils
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- CONFIG ----------------
BASE_DIR = "./results"
TFBS_MEME = './tfbs/20250424153556_JASPAR2024_combined_matrices_735317_meme.txt'
SEL_PPM_CSV = './tfbs/selected_ppms.csv'
# ----------------------------------------

# Regex to capture algo, dataset, and round
pattern = re.compile(rf"{re.escape(BASE_DIR)}/([^/]+)/([^_]+)_hard_\d+\.csv")

def find_groups():
    csv_files = glob.glob(os.path.join(BASE_DIR, "*", "*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {BASE_DIR}")

    grouped = defaultdict(list)
    for f in csv_files:
        m = pattern.match(f)
        if not m:
            continue
        alg, ds = m.group(1), m.group(2)
        # decide promoter vs enhancer
        suffix = "enhancers" if ds in ['hepg2', 'k562', 'sknsh'] else "promoters"
        key = f"{alg}_{ds}_{suffix}"
        grouped[key].append(f)

    return grouped

def analyze_group(key, files):
    # parse out algo, cell_type, data_type
    algo, cell_type, data_type = key.split('_')
    tasks = (['hepg2','k562','sknsh'] if data_type=="enhancers"
             else ['JURKAT','K562','THP1'])

    # 1) Core metrics summary
    records = []
    all_results = {}
    for f in files:
        res = process_file(f, tasks=tasks, col='true_score')
        all_results[f] = res
        for task, metrics in res.items():
            rec = {'file': f, 'task': task}
            rec.update(metrics)
            records.append(rec)

    df = pd.DataFrame(records)
    num_cols = ['top','fitness','mean','median','max','p95','diversity','novelty','violation_rate']
    summary = df.groupby('task')[num_cols].agg(['mean','std'])

    # 2) Tradeoff summary
    tradeoff_records = []
    for f, task_dict in all_results.items():
        tmp = []
        for task, metrics in task_dict.items():
            r = {'task': task}
            r.update(metrics)
            tmp.append(r)
        df_grp = pd.DataFrame(tmp)
        score = compute_tradeoff_score(df_grp, cell_type, alpha=1.0)
        tradeoff_records.append({'task': cell_type, 'tradeoff': score})

    df_t = pd.DataFrame(tradeoff_records)
    tradeoff_summary = df_t.groupby('task')['tradeoff'].agg(['mean','std'])

    # 3) Positional entropy
    pe_vals = []
    for f in files:
        data = pd.read_csv(f)
        seqs = data.loc[data['round']==100, 'sequence'].tolist()
        pe_vals.append(positional_entropy(seqs))
    pe_mean, pe_std = np.mean(pe_vals), np.std(pe_vals)

    # 4) GC divergence
    # load ground truth sequences once
    path = './human_promoters/lm_data/train1.csv' if data_type == 'promoters' else './data/human_enhancers/01_data_processing/lm_data/train.csv'
    gt_seqs = pd.read_csv(path).sequence.tolist()
    gc_kl, gc_w1 = [], []
    for f in files:
        data = pd.read_csv(f)
        seqs = data.loc[data['round']==100, 'sequence'].tolist()
        kl, w1 = gc_distance(gc_content(gt_seqs), gc_content(seqs))
        gc_kl.append(kl)
        gc_w1.append(w1)
    kl_mean, kl_std = np.mean(gc_kl), np.std(gc_kl)
    w1_mean, w1_std = np.mean(gc_w1), np.std(gc_w1)

    # 5) Motif correlation
    motifs, bg = scripts.motifs.read_meme(TFBS_MEME)
    sel = scripts.utils.load_csv(SEL_PPM_CSV).Matrix_id.tolist()
    motifs = [m for m in motifs if m.name.decode() in sel]

    corr_vals = []
    freq_path = f'./tfbs/{cell_type}_tfbs_freq_all.csv'
    for f in files:
        corr, _, _ = compute_motif_corr(motifs, bg, freq_path, f)
        corr_vals.append(corr)
    corr_mean, corr_std = np.mean(corr_vals), np.std(corr_vals)


    tradeoff_mean = tradeoff_summary.iloc[0]['mean']
    tradeoff_std = tradeoff_summary.iloc[0]['std']

    results = pd.DataFrame({
        'algo': [algo],
        'cell_type': [cell_type],
        'tradeoff': [f"{tradeoff_mean:.4f}({tradeoff_std:.4f})"],
        'positional_entropy': [f"{pe_mean:.4f}({pe_std:.4f})"],
        'gc_kl': [f"{kl_mean:.4f}({kl_std:.4f})"],
        'gc_w1': [f"{w1_mean:.4f}({w1_std:.4f})"],
        'motif_corr': [f"{corr_mean:.4f}({corr_std:.4f})"]
    })

    means = summary.xs('mean', axis=1, level=1)
    stds  = summary.xs('std',  axis=1, level=1)
    formatted = means.round(3).astype(str) + ' (' + stds.round(3).astype(str) + ')'



    return formatted, results

def main():
    grouped = find_groups()

    #print all groups
    print("=== Groups found ===")
    for key in grouped.keys():
        print(key)
    print("\n====================\n")
    for key, files in grouped.items():
        print(f"\n=== Group: {key} ===")
        summary, res = analyze_group(key, files)

        print(summary)
        print(res)
        # save summary to csv
        summary.to_csv(os.path.join(BASE_DIR,  f"{key}_summary.csv"))
        res.to_csv(os.path.join(BASE_DIR, f"{key}_results.csv"), index=False)

if __name__ == "__main__":
    main()