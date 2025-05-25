import numpy as np
import torch

class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, max_size, priority=True):
        self.memory = []
        self.max_size = max_size
        self.priority = priority

    def add_experience(self, dnas, obs, logprobs,scores, rewards,nonterms, episode_lens):
        if self.max_size <= 0:
            return
        
        obs = obs.T # (batch, seq_len)
        nonterms = nonterms.T # (batch, seq_len)
        rewards = rewards.permute(2, 0, 1) # (batch, seq_len)
        episode_lens = episode_lens
        logprobs = logprobs.T
        

        experience = zip(dnas, obs, logprobs,scores, rewards,nonterms, episode_lens)
        self.memory.extend(experience)

        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, dna_list = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in dna_list:
                    idxs.append(i)
                    # smiles.append(exp[0])
                    dna_list.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]

            #print(self.memory[3])
            self.memory.sort(key=lambda x: x[3], reverse=True)
            self.memory = self.memory[:self.max_size]
            
            
    def sample(self, n, device):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[3]+1e-10 for x in self.memory]
            #scores = [x[3] for x in sample]
            #prefs = [x[7] for x in sample]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]

            obs = [x[1] for x in sample]
            logprobs = [x[2] for x in sample]
            scores = [x[3] for x in sample]
            rewards = [x[4] for x in sample]
            nonterms = [x[5] for x in sample]
            lens = [x[6] for x in sample]
            
        obs = torch.stack(obs).transpose(0,1) # (seq_len, batch)
        nonterms = torch.stack(nonterms).transpose(0,1) # (seq_len, batch)
        scores = torch.tensor(scores, dtype=torch.float32, device=device) # (batch)
        rewards = torch.stack(rewards).permute(1,2,0) # (seq_len, obj,batch)
        lens = torch.stack(lens) 
        logprobs = torch.stack(logprobs).transpose(0,1)
        
 
        return obs, logprobs,scores, rewards, nonterms, lens

    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory.clear()