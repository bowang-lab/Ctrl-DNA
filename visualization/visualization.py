import numpy as np
import pandas as pd
import torch
import sys, os
# sys.path.append("")
sys.path.append('../src')
import reglm.lightning
sys.path.append('../')

import scripts.utils
from polygraph.sequence import min_edit_distance
import numpy as np
import pandas as pd
import torch
import sys, os, re
import reglm.lightning, reglm.regression

enhancer1 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_regression_paired_hepg2.ckpt')

enhancer2 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_regression_paired_k562.ckpt')

enhancer3 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_regression_paired_sknsh.ckpt')

promoter1 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_paired_jurkat.ckpt')

promoter2 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_paired_K562.ckpt')

promoter3 = reglm.regression.EnformerModel.load_from_checkpoint(
    '../ckpt/human_paired_THP1.ckpt')

enhancer = {
    'hepg2': enhancer1,
    'k562': enhancer2,
    'sknsh': enhancer3
}
promoter = {
    'JURKAT': promoter1,
    'K562': promoter2,
    'THP1': promoter3
}

import matplotlib.pyplot as plt
#plt reset
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8

rename_dict = {
    "log_barrier": "Log Barrier",
    "ours":        "Ours",
    "bo":          "BO",
    "TACO":        "TACO",
    "cmaes":       "CMA-ES",
    "lag_ppo_results": "Lag PPO",
    "adalead":     "AdaLead",
    "pex":         "PEX",
}
label_list = ["Ours", "Log Barrier", "BO", "TACO", "CMA-ES", "Lag PPO", "AdaLead", "PEX"]

COLOR_MAP = {
    "Ours" : "C0",
    "Log Barrier" : "C1",
    "BO" : "C2",
    "TACO" : "C3",
    "CMA-ES" : "C4",
    "Lag PPO" : "C5",
    "AdaLead" : "C6",
    "PEX" : "C7",
}

def plot_four_metrics(
    dfs: dict,               
    dataset: str,           
    constraint: float=None,  
) -> None:

    # metrics = ["r1", "r2", "r3", "de"]
    if dataset == "hepg2":
        metrics = ["r1", "r2", "r3", "de"]
        task_list = ["hepg2", "k562", "sknsh"]
    elif dataset == "k562":
        metrics = ["r2", "r1", "r3", "de"]
        task_list = ["k562", "hepg2", "sknsh"]
    elif dataset == "sknsh":
        metrics = ["r3", "r1", "r2", "de"]
        task_list = ["sknsh", "hepg2", "k562"]
    elif dataset == "JURKAT":
        metrics = ["r1", "r2", "r3", "de"]
        task_list = ["JURKAT", "K562", "THP1"]
    elif dataset == "K562":
        metrics = ["r2", "r1", "r3", "de"]
        task_list = ["K562", "JURKAT", "THP1"]
    elif dataset == "THP1":
        metrics = ["r3", "r1", "r2", "de"]
        task_list = ["THP1", "JURKAT", "K562"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 3), sharex=True)

    for id, (ax, metric) in enumerate(zip(axes, metrics)):
        if metric not in dfs:
            ax.set_visible(False)
            continue

        df = dfs[metric].rename(columns=rename_dict)

        for model in label_list:
            mean = df[model]["mean"]
            std  = df[model]["std"]

            if model == "Ours":
                ax.plot(
                    mean.index,
                    mean,
                    label=model,
                    color=COLOR_MAP[model],
                    linewidth=1.5,
                )
                ax.fill_between(
                    mean.index,
                    mean - 1 * std,
                    mean + 1 * std,
                    color=COLOR_MAP[model],
                    alpha=0.25,
                )
            else:
                ax.plot(
                    mean.index,
                    mean,
                    label=model,
                    color=COLOR_MAP[model],
                    linewidth=1,
                    alpha=0.8,
                )
                ax.fill_between(
                    mean.index,
                    mean - 1 * std,
                    mean + 1 * std,
                    color=COLOR_MAP[model],
                    alpha=0.15,
                )

        if id != 0 and metric != "de":
            ax.axhline(y=constraint, color="black", linestyle="--", linewidth=1)
            indicator = "↓"
        else:
            indicator = "↑"

        ax.set_xlabel("Round")
        if id<=2:
            ax.set_title(task_list[id] + " Fitness" + indicator)
            # ax.set_ylabel("Fitness")
        else:
            ax.set_title("DE" + indicator)
            # ax.set_ylabel("DE")

        # if metric == "de":
        #     ax.set_ylim(-0.5, 0.5)
        # else:
        #     ax.set_ylim(0, 1)

        ax.set_xlim(0, 100)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=dataset,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout()
    plt.savefig(f"./visualize_results/{dataset}.png", dpi = 300, bbox_inches='tight')
    plt.show()

def get_fitness_info(cell):
    if cell == 'hepg2':
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
    return length, min_fitness, max_fitness

result_dir = './baseline_results'

model_list = os.listdir(result_dir)
dataset_list = ["hepg2", "k562", "sknsh", "JURKAT", "K562", "THP1"]
# dataset_list = ["JURKAT", "K562", "THP1"]

for dataid, dataset in enumerate(dataset_list):
    if dataid < 3:
        is_enhancer = True
        task_list = dataset_list[:3]
    else:
        is_enhancer = False
        task_list = dataset_list[3:]

    # # task_list = dataset_list[:3] if is_enhancer else dataset_list[3:]
    # task_list = dataset_list
    # # dataset = dataset_list[0]
    # # model = model_list[7]
    # r1_dict = {}
    # r2_dict = {}
    # r3_dict = {}
    # de_dict = {}
    # for model in model_list:
    #     data_dir = os.path.join(result_dir, model)

    #     print(model, dataset)

    #     per_seed_means = []

    #     file_list = os.listdir(data_dir)
    #     pattern = r'^(?!.*summary).*\.csv$'
    #     for fname in file_list:
    #         match = re.match(pattern, fname)
    #         if match and fname.split("_")[0] == dataset:
    #             df = pd.read_csv(os.path.join(data_dir, fname))
    #             ds = reglm.regression.SeqDataset(df.sequence.tolist())
    #             oracle = enhancer if is_enhancer else promoter
    #             for k,v in oracle.items():
    #                 length, min_fitness, max_fitness = get_fitness_info(k)
    #                 df[k] = (v.predict_on_dataset(ds, batch_size=10000, device=0).squeeze()-min_fitness)/ (max_fitness-min_fitness)
    #             de = df[dataset]
    #             for i in task_list:
    #                 if i == dataset:
    #                     continue
    #                 de = de - df[i]/2
    #             df['de'] = de
    #             means = (
    #                 df[task_list+['de', 'round']]
    #                 .groupby("round", as_index=False)
    #                 .mean(numeric_only=True)
    #                 .set_index("round")
    #                 .sort_index()
    #             )
    #             per_seed_means.append(means)
        
    #     stacked = (
    #         pd.concat(per_seed_means, keys=range(len(per_seed_means)), names=["seed"])
    #         .reset_index(level="seed")
    #     )
        
    #     round_df = (
    #         stacked
    #         .groupby("round")
    #         .agg(["mean", "std"])      
    #         .reindex(range(1, 101))   
    #     )

    #     r1_dict[model] = round_df[task_list[0]]
    #     r2_dict[model] = round_df[task_list[1]]
    #     r3_dict[model] = round_df[task_list[2]]
    #     de_dict[model] = round_df['de']

    # r1 = pd.concat(r1_dict, axis=1)
    # r2 = pd.concat(r2_dict, axis=1)
    # r3 = pd.concat(r3_dict, axis=1)
    # de = pd.concat(de_dict, axis=1)

    # r1.to_csv(f"./visualize_results/{dataset}_{task_list[0]}.csv")
    # r2.to_csv(f"./visualize_results/{dataset}_{task_list[1]}.csv")
    # r3.to_csv(f"./visualize_results/{dataset}_{task_list[2]}.csv")
    # de.to_csv(f"./visualize_results/{dataset}_de.csv")

    r1 = pd.read_csv(f"./visualize_results/{dataset}_{task_list[0]}.csv", index_col=0, header=[0, 1])
    r2 = pd.read_csv(f"./visualize_results/{dataset}_{task_list[1]}.csv", index_col=0, header=[0, 1])
    r3 = pd.read_csv(f"./visualize_results/{dataset}_{task_list[2]}.csv", index_col=0, header=[0, 1])
    de = pd.read_csv(f"./visualize_results/{dataset}_de.csv", index_col=0, header=[0, 1])

    dfs = {
        "r1": r1,  
        "r2": r2,
        "r3": r3,
        "de": de,
    }

    plot_four_metrics(dfs, dataset=dataset, constraint=0.5)