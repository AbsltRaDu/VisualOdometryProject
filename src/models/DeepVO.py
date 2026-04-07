from collections import deque

import torch
import torch.nn as nn 
import torchvision.models as models


class convBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            )

    def forward(self, x):
        return self.block(x)

class CNNencoder(nn.Module): 
    
    def __init__(self, feet_dim=1024, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            convBlock(12, 64, (3, 3), 2, 1, dropout),
            convBlock(64, 128, (3, 3), 2, 1, dropout),
            convBlock(128, 256, (3, 3), 2, 1, dropout),
            convBlock(256, 256, (3, 3), 1, 1, dropout),
            convBlock(256, 512, (3, 3), 2, 1, dropout),
            convBlock(512, 512, (3, 3), 1, 1, dropout),
            convBlock(512, 512, (3, 3), 1, 1, dropout),
            convBlock(512, feet_dim, (3, 3), 2, 1, dropout))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.encoder(x) # (B, C, H, W)
        x = self.pool(x) # (B, C, 1, 1)
        return x.flatten(1) # (B, feet_dim)

class PairwiseVOModel(nn.Module): # Модель для обучения экодера. Учим его распознавать паттерны
    def __init__(self, feet_dim=1024, pose_dim=6, dropout=0.3):
        super().__init__()
        self.encoder = CNNencoder(feet_dim, dropout)
        self.fc = nn.Sequential( 
            nn.Linear(feet_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, pose_dim)
        )
        
    def forward(self, x):
        x = self.encoder(x) # (B, feet_dim)
        return self.fc(x) # (B, 6)

class DeepVORNN(nn.Module):
    def __init__(self, feat_dim=1024, hidden_size=64, num_layers=2, pose_dim=6, dropout=0.3):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, pose_dim)
        )
        
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        y = self.fc(out[:, -1, :])
        return y

class DeepVO:
    def __init__(self, encoder, rnn_model, seq_len, device):
        self.encoder = encoder.to(device).eval()
        self.rnn_model = rnn_model.to(device).eval()
        self.seq_len = seq_len
        self.device = device
        self.buffer = deque(maxlen=seq_len)
        
    @torch.no_grad() # Вырубаем градиенты для шага
    def step(self, x_pair):
        
        self.encoder.eval()
        self.rnn_model.eval()
        
        x_pair = x_pair.to(self.device) # (B, C, H, W)
        x = self.encoder(x_pair).cpu() # (B, C)
        self.buffer.append(x) # [(B, c)]
        
        if len(self.buffer) < self.seq_len:
            n = self.seq_len - len(self.buffer)
            z = torch.zeros((n, x.shape[0], x.shape[1])).to(self.device)
            x_seq = torch.stack((list(self.buffer)), dim=0).to(self.device) # (seq, B, C) -> (B, seq, C)
            x_seq = torch.vstack([z, x_seq]).transpose(0, 1)
            y = self.rnn_model(x_seq) # (1, 6)

            return y.squeeze(0).cpu() # (6)
        else:
            x_seq = torch.stack((list(self.buffer)), dim=0).transpose(0, 1).to(self.device) # (seq, B, C) -> (B, seq, C)
            y = self.rnn_model(x_seq) # (1, 6)

            return y.squeeze(0).cpu() # (6)
