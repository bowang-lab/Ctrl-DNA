import os
import sys
import wandb
import hydra
import torch
from torch import optim
import numpy as np
import pandas as pd

import src.reglm.regression

from .base_optimizer import BaseOptimizer, evaluate,BaseOptimizerMulti
import src.reglm.dataset, src.reglm.lightning, src.reglm.utils, src.reglm.metrics
import random
from copy import deepcopy
import scripts.utils, scripts.motifs
import scipy
def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)
BASES = ['A', 'C', 'G', 'T']

def mutate(seq, mutation_rate=0.01):
    """Randomly mutate a DNA sequence."""
    return ''.join(
        base if random.random() > mutation_rate else random.choice(BASES)
        for base in seq
    )

def crossover(seq1, seq2):
    """Single-point crossover between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length for crossover.")
    point = random.randint(1, len(seq1) - 1)
    return seq1[:point] + seq2[point:], seq2[:point] + seq1[point:]


def get_advantages(scores,d=0):
    scores = scores-d
    mean_grouped_rewards = scores.view(-1, scores.shape[-1]).mean(dim=-1,keepdim=True)
    std_grouped_rewards = scores.view(-1, scores.shape[-1]).std(dim=-1,keepdim=True)
    #print('+++++reward++++ mean std ',scores.shape,mean_grouped_rewards.shape,std_grouped_rewards.shape)
    advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    return advantages

def get_constraint_advantages(scores,d=0):
    d = torch.tensor(d)
    mean_grouped_rewards = scores.view(-1, scores.shape[-1]).mean(dim=-1,keepdim=True)
    std_grouped_rewards = scores.view(-1, scores.shape[-1]).std(dim=-1,keepdim=True)
    #print('+++++reward++++ mean std ',scores.shape,mean_grouped_rewards.shape,std_grouped_rewards.shape)
    advantages = (d - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    return advantages

def get_per_token_loss(coef_1,coef_2,advantages):
    per_token_loss1 = coef_1 * advantages.unsqueeze(0)
    per_token_loss2 = coef_2 * advantages.unsqueeze(0)
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    return per_token_loss
def decode_bases(idxs):
    """
    Decode base indices back to nucleotide letters using self.base_stoi.

    Args:
        idxs (torch.LongTensor): Tensor of shape (N, L)

    Returns:
        sequences (List[str]): List of nucleotide sequences
    """
    # Build inverse mapping if not already done
    base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
    
    base_itos = {v: k for k, v in base_stoi.items()}
    sequences = []
    for row in idxs:
        bases = [base_itos[int(i)] for i in row]
        sequences.append(''.join(bases))
    return sequences     
class Lagrange_optimizer(BaseOptimizerMulti):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._init(cfg)

    def _init(self, cfg):
        self.prefix_label =  cfg.prefix_label
        
        
        if cfg.wandb_log:
            wandb.init(
                project=cfg.project_name,
                name=cfg.wandb_run_name,
            )
        self.agent = src.reglm.lightning.LightningModel(label_len = len(cfg.prefix_label))
        self.label = cfg.prefix_label
        self.agent.to(self.device)
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg.lr)

        self.vocab = list(self.agent.label_stoi.keys()) + list(self.agent.base_stoi.keys())

        self.beta = cfg.beta
        self.epsilon = cfg.epsilon
        self.ref_model = src.reglm.lightning.LightningModel() #deepcopy(self.agent)  

        self.predict = self.predict_enformer
        self.REF_UPDATE_FREQUENCY=20

        self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)
        
        self.lambda_1 = cfg.lambda_value[0]
        self.lambda_2 = cfg.lambda_value[1]

        self.lagrangian_multipliers = [torch.nn.Parameter(torch.as_tensor(self.lambda_1), requires_grad=True),torch.nn.Parameter(torch.as_tensor(self.lambda_2), requires_grad=True),torch.nn.Parameter(torch.as_tensor(cfg.tfbs_ratio), requires_grad=True) ]
        
        if cfg.optimizer =='Adam':
            self.lambda_optimizers = [torch.optim.Adam( [self.lagrangian_multipliers[i]], lr=cfg.lambda_lr, eps=1e-4 ) for i in range(len(self.lagrangian_multipliers))]
        else:
            self.lambda_optimizers = [torch.optim.SGD( [self.lagrangian_multipliers[i]], lr=cfg.lambda_lr, momentum=0.1 ) for i in range(len(self.lagrangian_multipliers))]
        self.constraint  =cfg.constraint
        print(cfg.constraint)
        self.task_reward_map = {
            'hepg2': 0,
            'k562': 1,
            'sknsh':2,
            'JURKAT':0,
            'K562':1,
            'THP1':2
            # add more tasks here if needed
        }
        self.task_idx=self.task_reward_map.get(cfg.task, 0)

        motifs, bg = scripts.motifs.read_meme(
            cfg.meme_path
        )
        print(f"Total motifs: {len(motifs)}")
        sel = scripts.utils.load_csv(
            cfg.ppms_path
        ).Matrix_id.tolist()
        self.motif2idx = {name: i for i, name in enumerate(sel)}
        self.idx2motif = {i: name for i, name in enumerate(sel)}

        motifs = [m for m in motifs if m.name.decode() in sel]
        print(f"Selected motifs: {len(motifs)}")

        data_dir='./data'
        if cfg.task in ['hepg2','k562','sknsh']:
            self.gt_freq = pd.read_csv(f'{data_dir}/human/tfbs/{cfg.task}_tfbs_freq_all.csv')
        else:
            self.gt_freq = pd.read_csv(f'{data_dir}/human_promoters/tfbs/{cfg.task}_tfbs_freq_all.csv')
        self.motifs = motifs
        self.bg = bg
        self.tfbs_ratio=cfg.tfbs_ratio
        self.tfbs_upper=cfg.tfbs_upper
        self.lambda_upper =cfg.lambda_upper
        
    def update_lambda( self, avg_epcost ):

        for id, epcost in enumerate(avg_epcost):

            lambda_loss = -self.lagrangian_multipliers[id] * (epcost - self.constraint[id])
            
            self.lambda_optimizers[id].zero_grad()
            lambda_loss.backward()
            self.lambda_optimizers[id].step()
            

        # Return the actual loss function only for debugging purposes
        return lambda_loss
    
    def get_motif(self,obs):
        dna_list = decode_bases(obs.T[:,4:])
        #print(dna_list[0])
        
        motifs = [m for m in self.motifs if m.name.decode() in self.gt_freq.columns]
        
        tfbs_sites = scripts.motifs.scan(dna_list, motifs, self.bg)
        
        freq_df = pd.pivot_table(
            tfbs_sites, values="start", index="SeqID", columns="Matrix_id", aggfunc="count"
        ).fillna(0)
        
        all_columns = freq_df.columns.union(self.gt_freq.columns)

        # Reindex both to the unified index and columns, fill missing with 0
        freq_df = freq_df.reindex(columns=all_columns, fill_value=0)
        gt_freq = self.gt_freq.reindex(columns=all_columns, fill_value=0)

        freq_df.drop(columns=['SeqID'], inplace=True)
        gt_freq.drop(columns=['SeqID'], inplace=True)
        
        correlations = torch.zeros(len(dna_list)).to(self.device)
        gt_vec=np.array(gt_freq.sum(0))
        
        for seq_id in freq_df.index:
            if seq_id in freq_df.index:
                freq_vec = np.array(freq_df.loc[seq_id])
                # If both vectors are all zeros, skip or set correlation to NaN
                if np.all(freq_vec == 0) and np.all(gt_vec == 0):
                    continue  # or correlations.append(np.nan)
                
                corr = scipy.stats.pearsonr(freq_vec, gt_vec)[0]
                correlations[int(seq_id)]=-corr
        return correlations
    def softmax( self, x, temperature=1, total=1):
        return (np.exp(x/temperature) / np.sum(np.exp(x/temperature), axis=0)) * total
    def update(self, obs, old_logprobs,rewards, nonterms, episode_lens,correlations,cfg, metrics, log,iteration,epoch):
        # obs shape: torch.Size([204, 280]), old_logprobs shape: torch.Size([203, 280]), rewards shape: torch.Size([203, 3, 280]), nonterms shape: torch.Size([204, 280]), episode_lens shape: torch.Size([280]), prefs shape: torch.Size([3, 280])
        self.agent.train()
        constraint_indices = [i for i in range(rewards.shape[1]) if i != self.task_idx]
        print('task id and constrain id: ',self.task_idx,constraint_indices)
        '''
        add lagrangian 
        '''
        #cost_batch = rewards[-1,:-1,:].mean(-1).unsqueeze(-1)
        
        cost_batch = rewards[-1, constraint_indices, :].mean(-1).unsqueeze(-1)
        
        if correlations!=None:
            self.update_lambda(torch.cat([cost_batch,correlations.mean(-1).unsqueeze(-1).unsqueeze(-1)],dim=0))
        else:
            self.update_lambda(cost_batch)
        
        #for cost in cost_batch: self.update_lambda(cost)
        for i in range(3): self.lagrangian_multipliers[i].data.clamp_( min=0 )

        
        lambdas = np.array([lag.detach().numpy() for lag in self.lagrangian_multipliers])
        
        # To force the normalized value without affecting the gradient, we exploit the 'clamp' function from torch
        for i in range(2): self.lagrangian_multipliers[i].data.clamp_( min=lambdas[i], max=self.lambda_upper)
        
        self.lagrangian_multipliers[2].data.clamp_( min=lambdas[2], max=self.tfbs_upper)
        
        scores = rewards[-1, self.task_idx]
        mean_grouped_rewards = scores.view(-1, scores.shape[-1]).mean(dim=-1,keepdim=True)
        std_grouped_rewards = scores.view(-1, scores.shape[-1]).std(dim=-1,keepdim=True)
       
        advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        if correlations!=None:
            correlations_adv = get_advantages(correlations)
       
        constraint_indices = [i for i in range(rewards.shape[1]) if i != self.task_idx]
        advantages_2 = get_advantages(rewards[-1, constraint_indices[0]])
        advantages_3 = get_advantages(rewards[-1, constraint_indices[1]])

        if correlations!=None:
            total_lambda = self.lagrangian_multipliers[0] + self.lagrangian_multipliers[1]+self.lagrangian_multipliers[2]
        else:
            total_lambda = self.lagrangian_multipliers[0] + self.lagrangian_multipliers[1]

        # Define a soft inverse weighting factor (so high lambda → low boost)
        boost = max(1,2+self.tfbs_upper - total_lambda)
        if correlations!=None:
            advantages=boost*advantages-self.lagrangian_multipliers[0]*advantages_2-self.lagrangian_multipliers[1]*advantages_3-self.lagrangian_multipliers[2]*correlations_adv
        else:
            advantages=boost*advantages-self.lagrangian_multipliers[0]*advantages_2-self.lagrangian_multipliers[1]*advantages_3
        
        logprobs=self.agent.sequences_log_probs(obs,nonterms)
        
        old_per_token_logps = old_logprobs.detach().to(logprobs.device)
        coef_1 = torch.exp(logprobs - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(0)
        per_token_loss2 = coef_2 * advantages.unsqueeze(0)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        
        
        if self.beta != 0.0:
            
            ref_per_token_logps = self.ref_model.sequences_log_probs(obs,nonterms)
            per_token_kl = (torch.exp(ref_per_token_logps - logprobs) - (ref_per_token_logps - logprobs) - 1)
            per_token_loss += self.beta * per_token_kl
        
        loss = (per_token_loss).sum() / nonterms[:-1].sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        
        if log:
            step = iteration * cfg.epoch + epoch 
            
            log_data = {
                "update/pgloss": loss.item(),
                "update/reward_mean": scores.mean().item(),
                "update/reward_std": scores.std().item(),
                "update/kl_loss": per_token_kl.mean().item() if self.beta != 0.0 else 0.0,
                "update/advantages": advantages.mean().item(),
                "update/iteration": iteration,
                "update/epoch": epoch,
                "update/step": step
            }
            
            for i in range(rewards.shape[1]):  # Iterate over objectives (n)
                log_data[f"update/reward_mean_obj_{i}"] = rewards[-1:,i].squeeze(0).mean().item()
                log_data[f"update/reward_std_obj_{i}"] = rewards[-1:,i].squeeze(0).std().item()
            log_data[f"update/multipler_{0}"] = self.lagrangian_multipliers[0].item()
            log_data[f"update/multipler_{1}"] = self.lagrangian_multipliers[1].item()
            log_data[f"update/multipler_{2}"] = self.lagrangian_multipliers[2].item()
            if cfg.tfbs:
                log_data[f"update/corr"] = (-correlations).mean().item()
            if cfg.epoch>1:
                wandb.log(log_data, step=step)
            else:
                wandb.log(log_data)
            
            print(log_data)

    def optimize(self, cfg, starting_sequences):

        # results' dataframe
        columns = ['round', 'sequence', 'true_score']
        df = pd.DataFrame(columns=columns)
        summary_columns = ["round", "top", "fitness", "diversity", "novelty"]
        summary_df = pd.DataFrame(columns=summary_columns)
        # 1 add inital experience
        init_seqs = starting_sequences['sequence'].tolist()
        init_tokens = self.agent.encode(starting_sequences['sequence'].tolist(), [self.prefix_label] * len(init_seqs), add_start=True)
        init_tokens = init_tokens.T.to(self.device)
       
        num_rewards = cfg.num_rewards  # New parameter, e.g., num_rewards = 3
        init_rewards = torch.zeros(cfg.max_len + len(self.prefix_label), num_rewards, len(init_seqs), device=self.device)
        
        init_scores = starting_sequences['target'].tolist()
        
        init_rewards = torch.tensor(starting_sequences['rewards'].tolist()).to(self.device)
        init_rewards = init_rewards.T.unsqueeze(0).repeat(cfg.max_len + len(self.prefix_label),1,1)
        
        init_nonterms = [False] * len(self.prefix_label) + [True] * (cfg.max_len) + [False] 
        
        init_nonterms = torch.tensor(init_nonterms, dtype=torch.bool)
        init_nonterms = init_nonterms.unsqueeze(-1).expand(-1, len(init_seqs)).to(self.device)
        init_lens = [cfg.max_len] * len(init_seqs)
        init_lens = torch.tensor(init_lens, dtype=torch.long)
        init_logprobs = self.agent.sequences_log_probs(init_tokens, init_nonterms).detach().cpu()
        self.experience.add_experience(init_seqs, init_tokens,init_logprobs, init_scores, init_rewards,init_nonterms, init_lens)
        
        if cfg.tfbs:
            correlations=self.get_motif(init_tokens)
        else:
            correlations = None
        for i in range(cfg.epoch):

            self.update(init_tokens, init_logprobs,init_rewards, init_nonterms, init_lens,correlations,cfg, dict(), log = False,epoch = 0,iteration = 0)
        
        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        print('Start training ... ')
        # while eval_strings < cfg.max_strings:
        for it in range(1, cfg.max_iter + 1):
            
            with torch.no_grad():
                # sample experience, generate from old policy
                
                labels = [self.prefix_label] * cfg.batch_size
                obs, rewards, nonterms, episode_lens = self.agent.get_data(labels, cfg.max_len)
                rewards = rewards.unsqueeze(-1).repeat(1, 1, num_rewards)

                
                old_logprobs=self.agent.sequences_log_probs(obs, nonterms).detach().cpu()

            dna_list = []
            for dna in obs.cpu().numpy().T:
                dna_seq = self.agent.decode(dna, ignore_num=len(self.prefix_label)+1)[0]
                assert len(dna_seq) == cfg.max_len
                dna_list.append(dna_seq)
                
            scores = np.array(self.predict(dna_list)) #reward oracle
            #print('predicted scores: ',scores.shape)
            scores_multi = torch.tensor(scores, dtype=torch.float32, device=self.device)

            # combine scores with preference
            
            
            task_idx = self.task_reward_map.get(cfg.task, 0)
            constraint_indices = [i for i in range(scores_multi.shape[1]) if i != task_idx]
            
            scores = scores_multi[:,task_idx]-(scores_multi[:,constraint_indices[0]]-self.constraint[0])+scores_multi[:,task_idx]-(scores_multi[:,constraint_indices[1]]-self.constraint[1])
            
            scores = scores.detach().cpu()
            train_steps += 1
            
            log = self.cfg.wandb_log
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                log = True
                metrics = dict()
                scores = np.array(scores)
                metrics['eval_strings'] = eval_strings
                metrics['mean_score'] = np.mean(scores)
                metrics['max_score'] = np.max(scores)
                metrics['min_score'] = np.min(scores)
                
            
                wandb.log(metrics)
            
            rewards[-1, :] += scores_multi
            rewards=rewards.transpose(-2,-1)
            # replay buffer
            if len(self.experience) > cfg.e_batch_size:
                e_obs, e_logprobs,e_scores, e_rewards, e_nonterms, e_episode_lens = self.experience.sample(cfg.e_batch_size, self.device)
                e_L, e_B = e_obs.shape
                L, B = obs.shape

                f_L = max(e_L, L)

                #f_obs = torch.zeros((f_L, cfg.batch_size + cfg.e_batch_size), dtype=torch.long, device=self.device)
                #f_nonterms = torch.zeros((f_L, cfg.batch_size + cfg.e_batch_size), dtype=torch.bool, device=self.device)

                f_obs = torch.zeros((f_L, B + e_B), dtype=torch.long, device=self.device)
                f_nonterms = torch.zeros((f_L, B + e_B), dtype=torch.bool, device=self.device)

                f_obs[:L, :B] = obs
                f_obs[:e_L, B:] = e_obs

                f_nonterms[:L, :B] = nonterms
                f_nonterms[:e_L, B:] = e_nonterms
                
                
                f_rewards = torch.cat([rewards, e_rewards], dim=-1)
                
                f_episode_lens = torch.cat([episode_lens, e_episode_lens])
                
                f_logprobs = torch.cat([old_logprobs,e_logprobs],dim=-1)
                

                if cfg.tfbs:
                    correlations=self.get_motif(f_obs)
                else:
                    correlations = None
                for i in range(cfg.epoch):
                    self.update(f_obs, f_logprobs,f_rewards, f_nonterms, f_episode_lens,correlations,cfg, metrics, log,it,i)
            else:
                if cfg.tfbs:
                    correlations=self.get_motif(obs)
                else:
                    correlations = None
                for i in range(cfg.epoch):
                    # import pdb; pdb.set_trace()
                    self.update(obs, old_logprobs,rewards, nonterms, episode_lens,correlations,cfg, metrics, log,it,i)
            
            self.experience.add_experience(dna_list, obs, old_logprobs,scores, rewards, nonterms, episode_lens)
            
            round_df = pd.DataFrame({'round': [it]*len(dna_list), 'sequence': dna_list, 'true_score': scores[:len(dna_list)]})
            num_rewards = scores_multi.shape[1]
            #print(scores_multi.shape)
            scores_multi=scores_multi[:len(dna_list),:]
            for i in range(num_rewards):
                round_df[f'reward_{i+1}'] = np.array(scores_multi[:, i].cpu())
            df = pd.concat([df, round_df], ignore_index=True)
            round_results = evaluate(round_df, starting_sequences)
            round_results['round'] = it
            
            round_results_df = pd.DataFrame([round_results])
            summary_df = pd.concat([summary_df, round_results_df], ignore_index=True)
            
            if cfg.wandb_log:
                #print('round results:',round_results)
                wandb.log(round_results)
            print(f"Round {it} finished")
            print(round_results)

           
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_name = f'{cfg.out_dir}/{cfg.task}_{cfg.level}_{cfg.seed}_{cfg.max_iter}_{cfg.epoch}_{cfg.beta}_lambda_{cfg.lambda_lr}_opt_{cfg.optimizer}_tfbs_{cfg.tfbs_ratio}_tfbs_upper_{cfg.tfbs_upper}_lambdavalue_{self.lambda_1}_{self.lambda_2}_cons_{self.constraint[0]}.csv'
        summary_name = f'{cfg.out_dir}/{cfg.task}_{cfg.level}_{cfg.seed}_{cfg.max_iter}_{cfg.epoch}_{cfg.beta}_lambda_{cfg.lambda_lr}_opt_{cfg.optimizer}_tfbs_{cfg.tfbs_ratio}_tfbs_upper_{cfg.tfbs_upper}_lambdavalue_{self.lambda_1}_{self.lambda_2}_cons_{self.constraint[0]}_summary.csv'
        df.to_csv(save_name, index=False)
        summary_df.to_csv(summary_name, index=False)
        
        print('max training string hit')
        wandb.finish()
        sys.exit(0)