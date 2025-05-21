import os
#import tdc
import itertools
import time
import yaml
import wandb
import torch
import numpy as np

import src.reglm.dataset, src.reglm.lightning, src.reglm.utils, src.reglm.metrics
from .experience import Experience

def evaluate(round_df, starting_sequences):
    data = round_df.sort_values(by='true_score', ascending=False).iloc[:128]
    
    top_fitness = data.iloc[:16]['true_score'].mean().item()
    median_fitness = data['true_score'].median().item()
    
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
    
    
def get_fitness_info(cell,oracle_type='paired'):
    if oracle_type=='paired' or oracle_type=='dlow':
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
    
def top_auc(buffer, top_n, finish, env_log_interval, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(env_log_interval, min(len(buffer), max_oracle_calls), env_log_interval):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += env_log_interval * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

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
class BaseOptimizerMulti:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.task = cfg.task
        self.max_oracle_calls = cfg.max_oracle_calls
        self.env_log_interval = cfg.env_log_interval
        
        self.dna_buffer = dict()
        self.mean_score = 0

        # Logging counters
        self.last_log = 0
        self.last_log_time = time.time()
        self.last_logging_time = time.time()
        self.total_count = 0
        self.invalid_count = 0
        self.redundant_count = 0
        self.oracle_type=cfg.oracle_type
        if cfg.wandb_log:
            wandb.init(
                project=cfg.project_name,
                name=cfg.wandb_run_name,
            )
        
        self.device = torch.device(cfg.device)
        self.experience = Experience(cfg.e_size, cfg.priority)
        
        # Load target models for multiple cell types
        if cfg.task in ['hepg2','k562','sknsh']:
            self.targets = {
                'hepg2': self.load_target_model('hepg2'),
                'k562': self.load_target_model('k562'),
                'sknsh': self.load_target_model('sknsh')
            }
        else:
            self.targets = {
                'JURKAT': self.load_target_model('JURKAT'),
                'K562': self.load_target_model('K562'),
                'THP1': self.load_target_model('THP1')
            }
        
        self.fitness_ranges = {cell: get_fitness_info(cell,self.oracle_type) for cell in self.targets.keys()}
    
    def load_target_model(self, cell):
        
        model_path = {
            
            'hepg2':f'/human/ckpt/human_regression_{self.oracle_type}_hepg2.ckpt',
            'k562': f"/human/ckpt/human_regression_{self.oracle_type}_k562.ckpt",
            'sknsh': f"/human/ckpt/human_regression_{self.oracle_type}_sknsh.ckpt",
            "JURKAT":f"/human/ckpt/human_{self.oracle_type}_jurkat.ckpt",
            "K562":f"/human/ckpt/human_{self.oracle_type}_k562.ckpt",
            "THP1":f"/human/ckpt/human_{self.oracle_type}_THP1.ckpt",


        }
        model = src.reglm.regression.EnformerModel.load_from_checkpoint(
            model_path[cell], map_location='cuda:0'
        ).to(self.device)
        model.eval()
        return model
    
    def normalize_target(self, score, cell):
        _, min_fitness, max_fitness = self.fitness_ranges[cell]
        return (score - min_fitness) / (max_fitness - min_fitness)
    
    @torch.no_grad()
    def score_enformer(self, dna):
        if len(self.dna_buffer) > self.max_oracle_calls:
            return 0
        
        scores = []
        for cell, model in self.targets.items():
            #raw_score = model([dna]).squeeze(0).item()
            raw_score = self.targets[cell]([dna]).squeeze(0).item()
            #scores[cell] = self.normalize_target(raw_score, cell)
            norm_score = self.normalize_target(raw_score, cell)
            scores.append(norm_score)
        
        # Compute reward as hepg2 - k562 - sknsh
        #print('multi score is :++++++++++++++++',scores)
        # TODO -- Hypervolume Uncertainty Region
        reward = 0.8*scores[0] - 0.1*scores[1] - 0.1*scores[2]
        
        if dna in self.dna_buffer:
            self.dna_buffer[dna][2] += 1
            self.redundant_count += 1
        else:
            #self.dna_buffer[dna] = [float(reward), len(self.dna_buffer) + 1, 1]
            self.dna_buffer[dna] = [torch.tensor(scores), reward,len(self.dna_buffer) + 1, 1]
        
        return self.dna_buffer[dna][0]
    
    def predict_enformer(self, dna_list):
        st = time.time()
        assert type(dna_list) == list
        self.total_count += len(dna_list)
        
        score_list = [self.score_enformer(dna) for dna in dna_list]
        
        if len(self.dna_buffer) % self.env_log_interval == 0 and len(self.dna_buffer) > self.last_log:
            self.sort_buffer()
            self.log_intermediate()
            self.last_log_time = time.time()
            self.last_log = len(self.dna_buffer)
        
        self.last_logging_time = time.time() - st
        self.mean_score = np.mean(score_list)
        return score_list
    def sort_buffer(self):
        
        self.dna_buffer = dict(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][1], reverse=True))
            
    def log_intermediate(self, dna=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.dna_buffer.items())[:100]
            dnas = [item[0] for item in temp_top100]
            scores = [item[1][1] for item in temp_top100]
            n_calls = self.max_oracle_calls
        
        else:
            if dna is None and scores is None:
                if len(self.dna_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 dna in buffer
                    temp_top100 = list(self.dna_buffer.items())[:100]
                    dnas = [item[0] for item in temp_top100]
                    scores = [item[1][1] for item in temp_top100]
                    n_calls = len(self.dna_buffer)
                else:
                    results = list(sorted(self.dna_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    dnas = [item[0] for item in temp_top100]
                    scores = [item[1][1] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                raise NotImplementedError
       
        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
