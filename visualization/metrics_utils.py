
import numpy as np
import pandas as pd
import sys
# sys.path.append('')

import pandas as pd
import itertools
import time
import yaml
import wandb
import torch
import numpy as np
# Cells and levels configuration
cells = ["hepg2", "k562", 'sknsh']
levels = ["mbo"]
tfbs_lambdas = [0.0, 0.01, 0.1]
selected_round = 50
import src.reglm.dataset, src.reglm.lightning, src.reglm.utils, src.reglm.metrics,src.reglm.regression
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os

import scripts.motifs
from sklearn.cluster import AgglomerativeClustering
from itertools import product
#from pymemesuite import fimo
from pymemesuite.common import Sequence
from pymemesuite.fimo import FIMO
import scipy
import src

import pandas as pd
import scripts.utils, scripts.motifs
from tqdm import tqdm
from scipy.stats import entropy, wasserstein_distance

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def evaluate(round_df, starting_sequences,col='true_score',top=16):
    data = round_df.sort_values(by=col, ascending=False).iloc[:128]
    
    top_fitness = data.iloc[:top][col].mean().item() #16
    median_fitness = data[col].median().item()
    
    seqs = data['sequence'].tolist()
    
    distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
    diversity = np.median(distances) if distances else 0.0
    
    inits = starting_sequences['sequence'].tolist()
    novelty_distances = [min(distance(seq, init_seq) for init_seq in inits) for seq in seqs]
    novelty = np.median(novelty_distances) if novelty_distances else 0.0
    
    return {
        'top': top_fitness,
        'fitness': median_fitness,
        'diversity': diversity,
        'novelty': novelty
    }
def distance(s1, s2):
    return sum([1 if i != j else 0 for i, j in zip(list(s1), list(s2))])

def diversity(seqs):
    divs = []
    for s1, s2 in itertools.combinations(seqs, 2):
        divs.append(distance(s1, s2))
    return sum(divs) / len(divs)

def mean_distance(seq, seqs):
    divs = []
    for s in seqs:
        divs.append(distance(seq, s))
    return sum(divs) / len(divs)


def get_fitness_info(cell,oracle_type='paired'):
    if oracle_type=='paired':
        if cell == 'complex':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'defined':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'hepg2':
            length = 200
            min_fitness = -6.051336
            max_fitness = 10.992575
        elif cell == 'k562':
            length = 200
            min_fitness = -5.857445
            max_fitness = 10.781755
        elif cell == 'sknsh':
            length = 200
            min_fitness = -7.283977
            max_fitness = 12.888308
        elif cell == 'JURKAT':
            length = 250
            min_fitness = -5.574782
            max_fitness =8.555965
        elif cell == 'K562':
            length = 250
            min_fitness = -4.088671
            max_fitness = 10.781755
        elif cell == 'THP1':
            length = 250
            min_fitness = -7.271035
            max_fitness = 6.797082
        else:
            raise NotImplementedError()
    elif oracle_type=='separate':
        if cell == 'complex':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'defined':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'hepg2':
            length = 200
            min_fitness = -4.943863
            max_fitness = 8.981514
        elif cell == 'k562':
            length = 200
            min_fitness = -5.648375
            max_fitness = 10.099778
        elif cell == 'sknsh':
            length = 200
            min_fitness = -5.653600
            max_fitness = 11.185259
        else:
            raise NotImplementedError()
    elif oracle_type=='all':
        if cell == 'complex':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'defined':
            length = 80
            min_fitness = 0
            max_fitness = 17
        elif cell == 'hepg2':
            length = 200
            min_fitness = -6.051336
            max_fitness = 10.992575
        elif cell == 'k562':
            length = 200
            min_fitness = -5.857445
            max_fitness = 10.781755
        elif cell == 'sknsh':
            length = 200
            min_fitness = -7.283977
            max_fitness = 12.888308
        else:
            raise NotImplementedError()

    return length, min_fitness, max_fitness
def normalize_target(score,task,oracle='paired'):
    _,min_fitness,max_fitness =get_fitness_info(task,oracle)
    return (score - min_fitness) / (max_fitness - min_fitness)
def score_enformer(dna,target,task,oracle):
        
    #print('task is ...',task)    
    score = target([dna]).squeeze(0).item()
    #print(dna,score)
    score = normalize_target(score,task,oracle)
    #print('after normalize,',score)
    # score = .oracle([dna]).squeeze(0).item()
    
    return score
def predict_enformer(dna_list,target,task,oracle):
    
    score_list = []
    for dna in dna_list:
        score_list.append(score_enformer(dna,target,task,oracle))
        #print(dna,.score_enformer(dna))
            
        
    #mean_score = np.mean(score_list)
    return score_list


def evaluate_celltype(
    round_df,
    starting_sequences,
    task,
    oracle='paired',
    col='true_score',
    constraint_threshold=0.5
):
    # Load the target model based on task
    if task == 'hepg2':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_regression_paired_hepg2.ckpt", map_location='cuda:0').to('cuda:0')
    elif task == 'k562':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_regression_paired_k562.ckpt", map_location='cuda:0').to('cuda:0')
    elif task == 'sknsh':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_regression_paired_k562.ckpt", map_location='cuda:0').to('cuda:0')  # Possibly a mistake – uses k562 model for sknsh
    elif task == 'JURKAT':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_paired_jurkat.ckpt", map_location='cuda:0').to('cuda:0')
    elif task == 'K562':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_paired_K562.ckpt", map_location='cuda:0').to('cuda:0')
    elif task == 'THP1':
        target = src.reglm.regression.EnformerModel.load_from_checkpoint(
            f"./ckpt/human_paired_THP1.ckpt", map_location='cuda:0').to('cuda:0')
    else:
        raise NotImplementedError(f"Unsupported task: {task}")

    # Sort all sequences by predicted or true score
    data_sorted = round_df.sort_values(by=col, ascending=False)
    seqs_all = data_sorted['sequence'].tolist()
    scores_all = predict_enformer(seqs_all, target, task, oracle)

    # Top 128 statistics
    seqs_top = seqs_all[:128]
    scores_top = scores_all[:128]
    top_fitness = np.mean(scores_top[:16])  # Mean of top 16
    fitness = np.median(scores_top)  # Median of top 128
    violations_top = np.array(scores_top) > constraint_threshold
    violation_rate = np.mean(violations_top)

    # Max and p95 scores from all sequences
    max_score = scores_all[0]
    p95_index = int(np.ceil(len(scores_all) * 0.05)) - 1
    p95_score = scores_all[p95_index]

    # Mean and median from all scores
    mean_fitness = np.mean(scores_all)
    median_fitness = np.median(scores_all)

    # Diversity: median pairwise distance among top 128
    diversity = np.median([
        distance(s1, s2) for s1, s2 in itertools.combinations(seqs_top, 2)
    ]) if len(seqs_top) > 1 else 0.0

    # Novelty: median minimum distance from top 128 to initial sequences
    inits = starting_sequences['sequence'].tolist()
    novelty = np.median([
        min(distance(seq, init_seq) for init_seq in inits)
        for seq in seqs_top
    ]) if inits else 0.0

    return {
        'top': top_fitness,             # Mean fitness of top 16 in top 128
        'fitness': fitness,             # Median fitness of top 128
        'mean': mean_fitness,           # Mean of all normalized scores
        'median': median_fitness,       # Median of all normalized scores
        'max': max_score,               # Max score in the whole set
        'p95': p95_score,               # Score at 95th percentile rank
        'diversity': diversity,         # Diversity among top 128
        'novelty': novelty,             # Novelty of top 128 vs initial pool
        'violation_rate': violation_rate  # Constraint violation rate in top 128
    }


# Plot results as a clustered bar chart
def extract_cluster_name(file_path):
    match = re.search(r'/([^/]+)_hard', file_path)
    return match.group(1) if match else file_path
def plot_results(all_results):
    data_list = []
    
    # Convert dictionary to DataFrame-friendly format
    for file, celltypes in all_results.items():
        file = extract_cluster_name(file)  # Assuming this function extracts a meaningful name
        for celltype, metrics in celltypes.items():
            for metric, value in metrics.items():
                data_list.append({"File": file, "Celltype": celltype, "Metric": metric, "Value": value})

    df = pd.DataFrame(data_list)

    # Set plot style
    sns.set(style="whitegrid")

    # Define metrics and create subplots
    metrics = ["top", "fitness", "diversity", "novelty"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for subplots
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        subset = df[df["Metric"] == metric]
        
        sns.barplot(x="File", y="Value", hue="Celltype", data=subset, dodge=True, ax=ax)
        ax.set_title(f"{metric.capitalize()}")
        ax.set_xlabel("CSV File")
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
        ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def process_file(file_path, tasks=['hepg2', 'k562', 'sknsh'],oracle='paired',col='true_score',n=100):
    df = pd.read_csv(file_path)
    
    # Filter data for rounds
    last_round_data = df[df['round'] == n]
    starting_seqs = df[df['round'] == 1]
    
    # Store results
    results = {}
    
    for task in tasks:
        score = evaluate_celltype(last_round_data, starting_seqs, task,col=col,oracle=oracle)
        results[task] = score  # Store score for each task
    
    return results



def compute_tradeoff_score(df_group, target_task, alpha=1.0):
    """
    Args:
        df_group (pd.DataFrame): DataFrame for one file/run.
        target_task (str): Task to maximize.
        alpha (float): Weight for constraint violation penalty.

    Returns:
        float: Tradeoff Score
    """
    
    target_fitness = df_group.loc[df_group['task'] == target_task, 'fitness'].values[0]
    neg_target_fitness = df_group.loc[df_group['task'] != target_task, 'fitness']
    tradeoff_score = target_fitness - neg_target_fitness.mean()
    # constraint_violation_rates = df_group.loc[df_group['task'] != target_task, 'violation_rate']
    # print(constraint_violation_rates)
    # mean_violation_rate = constraint_violation_rates.mean()
    # tradeoff_score = target_fitness - alpha * mean_violation_rate
    return tradeoff_score

import numpy as np
import pandas as pd
import sys
# sys.path.append('/h/chenx729/TACOSourceCode/')

import scripts.motifs
from sklearn.cluster import AgglomerativeClustering
from itertools import product
#from pymemesuite import fimo
from pymemesuite.common import Sequence
from pymemesuite.fimo import FIMO
import scipy

import pandas as pd
import scripts.utils, scripts.motifs
from tqdm import tqdm
from scipy.stats import entropy, wasserstein_distance

def covariance_matrix(x):
    return np.cov(x)


def frobenius_norm(cov, cov2):
    return np.sqrt(np.sum((cov - cov2)**2))

def motif_count(sequence_list,motifs,bg):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a dictionary containing the motif counts for all the sequences
    '''
    #motifs, motif_file = load_jaspar_database(path_to_database)
    #motifs, bg = scripts.motifs.read_meme(
    #path)
    motif_ids = []
    occurrence = []

    sequences = [
        Sequence(seq, name=f"seq_{i}".encode())
        for i, seq in enumerate(sequence_list)
    ]
    fimo = FIMO(both_strands=True,threshold=0.0001)
    for motif in (motifs):
        pattern = fimo.score_motif(motif, sequences, bg)
        motif_ids.append(motif.accession.decode())
        occurrence.append(len(pattern.matched_elements))
    
    motif_counts = dict(zip(motif_ids,occurrence))

    return motif_counts,occurrence

def enrich_pr(count_1,count_2):
	c_1 = list(count_1.values())
	c_2 = list(count_2.values())

	return scipy.stats.pearsonr(c_1,c_2)

def compute_motif_corr(motifs,bg,freq_df_file,generated_file,round =100):
    freq_df = pd.read_csv(freq_df_file)
    motifs = [m for m in motifs if m.name.decode() in freq_df.columns]
    data = pd.read_csv(generated_file)
    data=data[data['round'] == round]['sequence']
    # data.index = data.index.astype(str)
    sites = scripts.motifs.scan(list(data), motifs, bg)
    
    freq_df_gen = pd.pivot_table(
                sites, values="start", index="SeqID", columns="Matrix_id", aggfunc="count"
            ).fillna(0)
    all_columns = freq_df.columns.union(freq_df_gen.columns)
    
    # Reindex both to the unified index and columns, fill missing with 0
    freq_df_1_aligned = freq_df.reindex(columns=all_columns, fill_value=0)
    freq_df_2_aligned = freq_df_gen.reindex(columns=all_columns, fill_value=0)
    freq_df_1_aligned.drop(columns=['SeqID'], inplace=True)
    freq_df_2_aligned.drop(columns=['SeqID'], inplace=True)
    correlations = np.zeros(len(data))
    gt_vec=freq_df_1_aligned.sum(0)
    
    for seq_id in freq_df_2_aligned.index:
        freq_vec = np.array(freq_df_2_aligned.loc[seq_id])
        # If both vectors are all zeros, skip or set correlation to NaN
        if np.all(freq_vec == 0) and np.all(gt_vec == 0):
            continue  # or correlations.append(np.nan)
        
        corr = scipy.stats.pearsonr(freq_vec, gt_vec)[0]
        if np.isnan(corr):
            corr=0.0
        correlations[int(seq_id)]=corr
    
    return np.array(correlations).mean(),freq_df_1_aligned,freq_df_2_aligned
def positional_entropy(seqs):
    """Mean Shannon entropy across positions."""
    L = len(seqs[0])
    ent = []
    for i in range(L):
        freqs = np.array(list(Counter(s[i] for s in seqs).values()), dtype=float)
        p = freqs / freqs.sum()
        ent.append(-np.sum(p * np.log2(p)))
    return np.mean(ent)
def kmer_diversity(seqs, k=3):
    """Fraction of observed k-mers vs possible 4^k."""
    seen = set()
    for s in seqs:
        for i in range(len(s) - k + 1):
            seen.add(s[i:i+k])
    return len(seen) / 4**k
def one_hot_encode_seqs(seqs, alphabet='ACGT'):
    """
    Turn a list of equal-length strings over 'ACGT' into an
    (N, L, 4) one-hot array of dtype int8.
    """
    mapping = {nt: i for i, nt in enumerate(alphabet)}
    N = len(seqs)
    L = len(seqs[0]) if N else 0
    X = np.zeros((N, L, len(alphabet)), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, nt in enumerate(s):
            X[i, j, mapping[nt]] = 1
    return X

def gc_distance(gc_gt, gc_gen, eps=1e-10):
    # for JSD: discretize into bins
    bins = np.linspace(0,1,51)
    p_gt, _ = np.histogram(gc_gt, bins=bins, density=True)
    p_gen, _ = np.histogram(gc_gen, bins=bins, density=True)
    # add small ε to avoid zeros
    eps = 1e-8
    P = p_gt + eps; Q = p_gen + eps
    M = 0.5*(P+Q)
    jsd = 0.5*(entropy(P, M) + entropy(Q, M))
    w1  = wasserstein_distance(gc_gt, gc_gen)
    return jsd,w1
def gc_content(seqs):
    
    contents = []
    for s in seqs:
        s_up = s.upper()
        gc = s_up.count('G') + s_up.count('C')
        total = sum(s_up.count(b) for b in ('A','C','G','T'))
        contents.append(gc / total if total > 0 else 0.0)
    return contents