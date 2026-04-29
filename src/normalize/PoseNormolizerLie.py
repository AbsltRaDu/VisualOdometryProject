import json
import os

import torch

class PoseNormalizerLie:

    def __init__(self, eps=1e-8):
        self.eps = eps

    def fit(self, data):
        self.mean_t = data[..., :3].mean(0)
        self.std_t = data[..., :3].std(0)

        self.mean_r = data[..., 3:].mean(0)
        self.std_r = data[..., 3:].std(0)

    def normalize(self, data):

        t = (data[..., :3] - self.mean_t) / (self.std_t + self.eps)
        r = (data[..., 3:] - self.mean_r) / (self.std_r + self.eps)

        return torch.cat([t, r], dim=-1)

    def denormalize(self, data):

        t = data[..., :3] * (self.std_t + self.eps) + self.mean_t
        r = data[..., 3:] * (self.std_r + self.eps) + self.mean_r

        return torch.cat([t, r], dim=-1)
    
    def save(self, path, name='params_of_normalize.json'):
        
        dct = {
            'mean_t': self.mean_t.tolist(),
            'std_t': self.std_t.tolist(),
            'mean_r': self.mean_r.tolist(),
            'std_r': self.std_r.tolist()
        }
        
        with open(os.path.join(path, 'params_of_normalize.json'), 'w', encoding='utf-8') as f:
            json.dump(dct, f, ensure_ascii=False, indent=4)
            
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            dct = json.load(f)
            
            self.mean_t = torch.tensor(dct['mean_t'])
            self.std_t = torch.tensor(dct['std_t'])

            self.mean_r = torch.tensor(dct['mean_r'])
            self.std_r = torch.tensor(dct['std_r'])