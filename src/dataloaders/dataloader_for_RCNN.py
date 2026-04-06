import torch
import torch.nn as nn 
from torch.utils import data

class dataloaderRCNN(data.Dataset):
    def __init__(self, tensor_x, tensor_y, seq_length, groups):
        self.seq_length = seq_length
        self.X = tensor_x
        self.Y = tensor_y
        
        self.windows = []
        

        for group in groups:

            if len(group) < seq_length:
                continue

            for i in range(len(group) - seq_length + 1):
                window = group[i:i + seq_length]
                self.windows.append(window)
        
                
        self.length = len(self.windows)
              
    def __getitem__(self, item):
        windows = self.windows[item]
        x_seq = self.X[windows].clone()
        y = self.Y[windows[-1]].clone()
        return x_seq, y
    
    def __len__(self):
        return self.length