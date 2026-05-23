from pathlib import Path
import sys
from typing import Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.models as models


class DeepVO2D(nn.Module):
    def __init__(self, VisualEncoder, feat_dim=1024, hidden_size=1000, num_layers=2, dropout=0.3):
        
        '''
        RNN + FCx2 блока для возможности трансферного обучения
        '''
        super().__init__()
        
        self.VisualEncoder = VisualEncoder
        
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x_seq: torch.Tensor, hidden=None):
        
        if x_seq.ndim != 5:
            raise ValueError(f"x_seq должен иметь форму (B, S, 12, H, W), а получил {x_seq.shape}")
        
        B, S, C, H, W = x_seq.shape
        
        fused_features = []

        for t in range(S):
            
            x_t = x_seq[:, t]  # (B, 12, H, W)

            visual_feat = self.VisualEncoder(x_t)  # (B, visual_dim)
            
            fused_features.append(visual_feat)
        
        fused_seq = torch.stack(fused_features, dim=1) # Собираем последовательность признаков
        rnn_out, hidden = self.rnn(fused_seq, hidden) # Прогоняем последовательность через основную RNN

        p, r = self.fc1(rnn_out), self.fc2(rnn_out) # (B, S, 3)
        return torch.cat([p, r], dim=-1).to(dtype=torch.float64), hidden

    @staticmethod
    def detach_hidden(hidden):
        '''
        Отсоединяет hidden от предыдущего вычислительного графа
        
        по методу truncated BPTTКак 
        '''
        
        if hidden is None:
            return None
        
        if isinstance(hidden, tuple): # Для LSTM
            return tuple(h.detach() for h in hidden)
 
