from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):


    def __init__(
        self,
        input_channels: int = 6,
        input_res: Tuple[int, int] = (384, 1280),
        conv_dropout=0.1,
    ) -> None:
        super(CNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=conv_dropout)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()


        x = torch.zeros(1, input_channels, *input_res)
        self.features_dim = int(np.prod(self.forward(x).size()))

    def forward(self, x: Tensor) -> Tensor:

        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.dropout(self.relu(self.conv3_1(x)))
        x = self.dropout(self.relu(self.conv4(x)))
        x = self.dropout(self.relu(self.conv4_1(x)))
        x = self.dropout(self.relu(self.conv5(x)))
        x = self.dropout(self.relu(self.conv5_1(x)))
        x = self.dropout(self.conv6(x))
        x = self.flatten(x)
        return x


class DeepVO(nn.Module):


    def __init__(
        self,
        input_channels: int = 6,
        input_res: list = [192, 640],
        hidden_size: int = 1000,
        lstm_layers: int = 2,
        output_size: int = 6,
        lstm_dropout: float = 0.2,
        conv_dropout: float = 0.1,
    ) -> None:

        super(DeepVO, self).__init__()

        self.feature_extractor = CNN(
            input_channels=input_channels,
            input_res=input_res,
            conv_dropout=conv_dropout,
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_extractor.features_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self, x_seq: Tensor, hidden_state: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:

        
        if x_seq.ndim != 5:
            raise ValueError(f"x_seq должен иметь форму (B, S, 12, H, W), а получил {x_seq.shape}")
        
        B, S, C, H, W = x_seq.shape
        
        fused_features = []
        
        for t in range(S):
            
            x_t = x_seq[:, t]  # (B, 12, H, W)        

            features = self.feature_extractor(x_t)
            
            fused_features.append(features)
            
        fused_seq = torch.stack(fused_features, dim=1) # Собираем последовательность признаков
        features = fused_seq.view(B, S, -1)
        
        lstm_out, hidden_state = self.lstm(features, hidden_state)

        # Forward pass through fully connected layer to predict pose
        y = self.fc(lstm_out)

        return y.to(dtype=torch.float64), hidden_state

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


    
