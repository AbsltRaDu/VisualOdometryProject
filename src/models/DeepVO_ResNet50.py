import torch
import torch.nn as nn
import torchvision.models as models

from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO

class DeepVOResNet50(nn.Module):
    
    def __init__(self, encoder, feat_dim=1024, hidden_size=1000, num_layers=2, pose_dim=6, dropout=0.3):
        '''
        ecnoder: Модель свертки, архитектура, которой должна иметь в себе слой "encoder"
        '''
        super().__init__()

        self.encoder = encoder.encoder
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
        
        S = x.shape[1] # берем длину последовательности
        features = []
        
        for s in range(S):
            x_s = x[:, s] # (B, C, H, W)
            feat_s = self.encoder(x_s)
            features.append(feat_s)

        features = torch.stack(features, dim=1) # (B, S, F)
        rnn_out, hidden = self.rnn(features, hidden) 
        
        p, r = self.fc1(rnn_out), self.fc2(rnn_out)
        return torch.hstack([p, r]).to(dtype=torch.float64), hidden
    
    @staticmethod
    def detach_hidden(hidden):
        '''
        Отсоединяет hidden от предыдущего вычислительного графа
        '''
        
        if isinstance(hidden, tuple): # Для LSTM
            return tuple(h.detach() for h in hidden)