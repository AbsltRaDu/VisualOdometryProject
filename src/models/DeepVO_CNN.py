import torch
import torch.nn as nn
import torchvision.models as models

from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO

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


class DeepVO_CNN(nn.Module):
    
    def __init__(self, feat_dim=1024, hidden_size=1000, num_layers=2, dropout=0.3):
        '''
        ecnoder: Модель свертки, архитектура, которой должна иметь в себе слой "encoder"
        '''
        super().__init__()

        self.encoder  = nn.Sequential(
            convBlock(12, 64, (3, 3), 2, 1, dropout),
            convBlock(64, 128, (3, 3), 2, 1, dropout),
            convBlock(128, 256, (3, 3), 2, 1, dropout),
            convBlock(256, 256, (3, 3), 1, 1, dropout),
            convBlock(256, 512, (3, 3), 2, 1, dropout),
            convBlock(512, 512, (3, 3), 1, 1, dropout),
            convBlock(512, 512, (3, 3), 1, 1, dropout),
            convBlock(512, feat_dim, (3, 3), 2, 1, dropout))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        
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
            nn.Linear(256, 3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        
    def forward(self, x, hidden=None):
        
        if x.ndim != 5:
            raise ValueError(f"x должен иметь форму (S, S, C, H, W), а имеет {x.shape}")
        
        S = x.shape[1] # берем длину последовательности
        features = []
        
        for s in range(S):
            x_s = x[:, s] # (B, C, H, W)
            feat_s = self.encoder(x_s)
            feat_s = self.pool(feat_s)
            
            features.append(feat_s.flatten(1))

        features = torch.stack(features, dim=1) # (B, S, F)
        rnn_out, hidden = self.rnn(features, hidden) 
        
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