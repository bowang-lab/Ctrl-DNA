from dataclasses import dataclass

import torch
import numpy as np
import random
import pandas as pd
import argparse

from dna_optimizers_multi.base_optimizer import get_fitness_info
from dna_optimizers_multi.lagrange_optimizer import Lagrange_optimizer 
import os
os.environ["WANDB_SILENT"] = "true"

def get_meme_and_ppms_path(task):
    tfbs_dir = '/h/chenx729/TACOSourceCode/TACO/TFBS'
    if task == "hepg2" or task == "k562" or task == "sknsh" or task=='JURKAT' or task=='K562' or task =='THP1':
        meme_path = f"{tfbs_dir}/20250424153556_JASPAR2024_combined_matrices_735317_meme.txt"
        ppms_path = f"{tfbs_dir}/selected_ppms.csv"
    else:
        raise ValueError(f"Task {task} not supported.")
    return meme_path, ppms_path
def get_prefix_label(task, level):
    if task == "hepg2" or task =='JURKAT':
        if level == "hard":
            return "100"
        
    if task == "k562" or task=='K562':
        if level == "hard":
            return "010"
        
    if task == "sknsh" or task =='THP1':
        if level == "hard":
            return "001"
        
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for OptimizerArguments.")

    parser.add_argument("--task", type=str, default="complex", help="The task name.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type (e.g., 'cuda' or 'cpu').")

    parser.add_argument("--max_oracle_calls", type=int, default=4000000, help="Maximum number of oracle calls.")
    parser.add_argument("--max_strings", type=int, default=384000000, help="Maximum number of strings.")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs.")
    
    parser.add_argument("--wandb_log", action='store_true', help="Enable logging with wandb.")
    parser.add_argument("--train_log_interval", type=int, default=1, help="Training log interval.")
    parser.add_argument("--env_log_interval", type=int, default=256, help="Environment logging interval.")

    parser.add_argument("--level", type=str, default="hard", help="Difficulty level.")
    parser.add_argument("--e_size", type=int, default=100, help="Experience size.")
    parser.add_argument("--e_batch_size", type=int, default=24, help="Experience batch size.")
    parser.add_argument("--priority", type=bool, default=True, help="Use priority in experience replay.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tfbs_lambda", type=int, default=1, help="Lambda value for TFBS.")

    #GRPO
    parser.add_argument("--beta", type=float, default=0.01, help="Random seed.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Random seed.")
    parser.add_argument("--grpo", type = bool,default=False)
    parser.add_argument("--num_rewards", type = int,default=3)
    parser.add_argument("--top", type = int,default=10)
    parser.add_argument("--out_dir", type = str,default='results')
    parser.add_argument("--project_name", type = str,default='Hyena_RL')
    parser.add_argument('--oracle_type',default='separate')
    parser.add_argument('--lambda_lr',default=3e-1,type=float)
    parser.add_argument('--optimizer',default='Adam',type=str)
    parser.add_argument('--tfbs_ratio',default=0.01,type=float)
    parser.add_argument('--tfbs_upper',default=0.01,type=float)
    parser.add_argument('--lambda_value',default=[0.0,0.0],nargs='+', type=float)
    parser.add_argument('--constraint',default=[0.5,0.5,-0.5],nargs='+', type=float)
    parser.add_argument('--tfbs',default=False,type = bool)
    parser.add_argument('--lambda_upper',default=1.0,type = float)
    args = parser.parse_args()
    
    return args
    

def main():
    optimizer_args = parse_args()
    optimizer_args.prefix_label = get_prefix_label(optimizer_args.task, optimizer_args.level)
    
    print(optimizer_args)
    optimizer_args.max_len, min_fitness, max_fitness = get_fitness_info(optimizer_args.task,optimizer_args.oracle_type)
    set_seed(optimizer_args.seed)
    optimizer_args.wandb_run_name = (
        f"{optimizer_args.task}_{optimizer_args.level}_grpo_{optimizer_args.grpo}_"
        f"lr_{optimizer_args.lr}_"
        f"epoch_{optimizer_args.epoch}_"
        f"seed_{optimizer_args.seed}_"
        f"beta_{optimizer_args.beta}_"
        f"lambda_{optimizer_args.lambda_lr}_"
        f"optimizer_{optimizer_args.optimizer}_"
        f"tfbs_{optimizer_args.tfbs_ratio}_"
        f"tfbs_upper_{optimizer_args.tfbs_upper}_"
        f"lambdavalue_{optimizer_args.lambda_value[0]}_{optimizer_args.lambda_value[1]}_"
        f"tfbs_loss_{optimizer_args.tfbs}"
        f"constraint_{optimizer_args.constraint[0]}_{optimizer_args.constraint[1]}"
        
)

    print(f"Optimizer Arguments: {optimizer_args}")
    optimizer_args.meme_path, optimizer_args.ppms_path = \
        get_meme_and_ppms_path(optimizer_args.task)
    
    
    optimizer=Lagrange_optimizer(optimizer_args)
    
    
    data_dir='/scratch/ssd004/scratch/chenx729/GRPO/'
    init_data_path = f'{data_dir}/data/human/{optimizer_args.task}_{optimizer_args.level}.csv'

    if optimizer_args.task in ['hepg2','k562','sknsh']:
        init_data_path = f'{data_dir}/human/{optimizer_args.task}_{optimizer_args.level}.csv'
        #score_name = f'{optimizer_args.task}_mean'
        cell_types = ['hepg2', 'k562', 'sknsh']
    else:
        init_data_path=f'{data_dir}/human_promoters/rl_data_large/{optimizer_args.task}_{optimizer_args.level}.csv'
        #score_name = f'{optimizer_args.task}'
        cell_types = ['JURKAT', 'K562', 'THP1']

    starting_sequences = pd.read_csv(init_data_path)
    if optimizer_args.task in ['hepg2','k562','sknsh']:
        starting_sequences.columns = starting_sequences.columns.str.lower()
    # min-max normalize the true scores
    
    normed_scores = []  # will hold normalized series for each cell type
    print('before preprocessing...')
    print(starting_sequences)
    all_col_names=[]
    for cell in cell_types:
        # Get min and max fitness for this cell type.
        _, min_fitness, max_fitness = get_fitness_info(cell,optimizer_args.oracle_type)
        col_name = f"{cell}_mean"
        all_col_names.append(col_name)
        # Compute normalized score for this cell type.
        
        starting_sequences[col_name] = (starting_sequences[col_name] - min_fitness) / (max_fitness - min_fitness)
    #score_name = 'exp_complex'
    
    mean_cols = [col for col in starting_sequences.columns if col.endswith('_mean')]

    # Get the current task column
    score_name = f"{optimizer_args.task}_mean"

    # Get other cell types
    other_cell_types = [col for col in mean_cols if col != score_name]
    print(other_cell_types)
    
    
    starting_sequences['target']= starting_sequences[score_name]-(
            starting_sequences[other_cell_types[0]]-optimizer_args.constraint[0])+ starting_sequences[score_name]- (starting_sequences[other_cell_types[1]]-optimizer_args.constraint[1])
   
    starting_sequences['rewards'] = starting_sequences[all_col_names].values.tolist()
    starting_sequences = starting_sequences.sort_values('target', ascending=False).head(128)
    print(starting_sequences)
        
    optimizer.optimize(optimizer_args, starting_sequences)

if __name__ == "__main__":
    main()