from pathlib import Path
import sys
from typing import Optional
from types import SimpleNamespace

import torch 
import torch.nn as nn


class imuEncoder(nn.Module):
    
    def __init__(self, imu_dim: int = 6, hidden_size: int = 128, out_dim: int = 256):
        '''
        LSTM для imu
        '''
        
        super().__init__()
        self.out_dim = out_dim
        self.gru = nn.GRU(
            input_size=imu_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, self.out_dim)
        
        
    def forward(self, imu: torch.Tensor):
        
        _, h = self.gru(imu) # (B, N_imu, 6)
        h = h[-1] # (1, B, hidden_size) -> (B, hidden_size), т.е. вытаскиваем код всего окна
        feat = self.fc(h) # (B, out_dim)

        return feat

class ViNet(nn.Module):
    
    def __init__(self, VisualEncoder, imuEncoder, rnn_hidden: int = 512, dropout: float = 0.1):
        
        super().__init__()
        
        self.VisualEncoder = VisualEncoder
        self.imuEncoder = imuEncoder
        
        self.lstm = nn.LSTM(
            input_size=self.VisualEncoder.features_dim+self.imuEncoder.out_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(rnn_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        
    def forward(self, x_seq: torch.Tensor, imu_seq: torch.Tensor, hidden=None):
        '''
        x_seq:   (B, S, 12, H, W)
        imu_seq: (B, S, N_imu, 6)
        '''
        
        if x_seq.ndim != 5:
            raise ValueError(f"x_seq должен иметь форму (B, S, 12, H, W), а получил {x_seq.shape}")

        if imu_seq.ndim != 4:
            raise ValueError(f"imu_seq должен иметь форму (B, S, N_imu, 6), а получил {imu_seq.shape}")

        B, S, C, H, W = x_seq.shape

        fused_features = []

        for t in range(S):
            
            x_t = x_seq[:, t]  # (B, 12, H, W)
            imu_t = imu_seq[:, t]  # (B, N_imu, 6)

            visual_feat = self.VisualEncoder(x_t)  # (B, visual_dim)
    
            imu_feat = self.imuEncoder(imu_t)  # (B, imu_dim)
            
            
            fused = torch.cat([visual_feat, imu_feat], dim=1)
            fused_features.append(fused)

        
        fused_seq = torch.stack(fused_features, dim=1) # Собираем последовательность признаков

        
        rnn_out, hidden = self.lstm(fused_seq, hidden) # Прогоняем последовательность через основную RNN

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