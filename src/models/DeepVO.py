import torch
import torch.nn as nn
import torchvision.models as models

class DeepVO(nn.Module):
    def __init__(self, feat_dim=1024, hidden_size=1000, num_layers=2, dropout=0.3):
        
        '''
        RNN + FCx2 блока для возможности трансферного обучения
        '''
        
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
        
    def forward(self, x, hidden):
        
        rnn_out, hidden = self.rnn(x, hidden) 
    
        p, r = self.fc1(rnn_out), self.fc2(rnn_out) # (B, S, 3)
        return torch.cat([p, r], dim=-1), hidden
        
 
class DeepVO_CNN(nn.Module):
    
    def __init__(self, encoder, feat_dim=1024, hidden_size=1000, num_layers=2, dropout=0.3):
        '''
        ecnoder: Модель свертки, архитектура, которой должна иметь в себе слой "encoder"
        '''
        super().__init__()

        self.encoder = encoder
        
        self.head = DeepVO(feat_dim, hidden_size, num_layers, dropout)
        
    def forward(self, x, hidden=None):
        
        if x.ndim != 5:
            raise ValueError(f"x должен иметь форму (S, S, C, H, W), а имеет {x.shape}")
        
        S = x.shape[1] # берем длину последовательности
        features = []
        
        for s in range(S):
            x_s = x[:, s] # (B, C, H, W)
            feat_s = self.encoder(x_s)
            features.append(feat_s)

        features = torch.stack(features, dim=1) # (B, S, F)        
        p, r = self.head(features, hidden)
        
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
        
class PairwiseVOModel(nn.Module): # Модель для обучения экодера. Учим его распознавать паттерны
    def __init__(self, encoder, feet_dim=1024, pose_dim=6, dropout=0.3):
        '''
        Обертка для отдельного обучения энкодера, который потом будет интегрирован в DeepVO
        '''
        
        
        
        super().__init__()
        
        self.encoder = encoder
        
        self.fc1 = nn.Sequential(
            nn.Linear(feet_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feet_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
        
    def forward(self, x):
        x = self.encoder(x) # (B, feet_dim)
        
        p, r = self.head(x)
        
        return torch.cat([p, r], dim=-1).to(dtype=torch.float64)