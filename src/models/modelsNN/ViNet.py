from pathlib import Path
import sys
from typing import Optional
from types import SimpleNamespace

import torch 
import torch.nn as nn

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[4]
WORKSPACE_ROOT = PROJECT_ROOT.parent
FLOWNET_ROOT = WORKSPACE_ROOT / 'flownet2-pytorch'
sys.path.append(str(FLOWNET_ROOT))

PROJECT_ROOT = Path.cwd()
FLOWNET_ROOT = PROJECT_ROOT.parent / 'flownet2-pytorch'

if not FLOWNET_ROOT.exists():
    raise FileNotFoundError(f'Не найден репозиторий FlowNet2: {FLOWNET_ROOT}')

# Проверяем, что есть нужный файл.
if not (FLOWNET_ROOT / 'networks' / 'FlowNetS.py').exists():
    raise FileNotFoundError(f"Не найден FlowNetS.py в: {FLOWNET_ROOT / 'networks'}")

sys.path.append(str(FLOWNET_ROOT))

from networks.FlowNetS import FlowNetS

class VisualEncoder(nn.Module):
    
    def __init__(self, checkpoint_path: Optional[str] = None, out_dim: int = 1024, freeze: bool = True, batch_norm: bool = False):
        '''
        энкодер ViNet на базе модели NVIDIA FlowNetS
        '''
        
        super().__init__()
        
        
        self.out_dim = out_dim
        args = SimpleNamespace()
        
        self.flownets = FlowNetS(
            args=args,
            input_channels=6,
            batchNorm=batch_norm
        )
        
        if checkpoint_path is not None:
            self._load_pretrained_weights(checkpoint_path)
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 1024, H, W) -> (B, 1024)
        self.fc = nn.Linear(1024, self.out_dim)
        
        if freeze:
            for param in self.flownets.parameters():
                param.requires_grad = False
    
    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        '''
        Загружает веса из NVIDIA checkpoint.
        '''

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_key = new_key.replace('flownets.', '')
            cleaned_state_dict[new_key] = value

        missing, unexpected = self.flownets.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        print(f'[FlowNet2SEncoder] Missing keys: {len(missing)}')
        print(f'[FlowNet2SEncoder] Unexpected keys: {len(unexpected)}')

    
    def _normalize_like_flownet(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Нормализация в стиле FlowNet2S.

        NVIDIA FlowNet2S обычно работает с изображениями в диапазоне примерно [0, 255]
        '''
        
        if x.max() <= 2.0:
            x = x * 255.0

        # Считаем среднее отдельно для каждого элемента батча и канала 
        rgb_mean = x.contiguous().view(x.size(0), x.size(1), -1).mean(dim=-1)
        rgb_mean = rgb_mean.view(x.size(0), x.size(1), 1, 1)

        
        x = (x - rgb_mean) / 255.0 # Центрируем и масштабируем

        return x
    
    
    def forward(self, x: torch.Tensor):
        '''
        (B, 12, H, W)
        '''
        
        if x.ndim != 4:
            raise ValueError(f"x должен иметь форму (B, 12, H, W), а получил {x.shape}")

        if x.shape[1] != 6:
            raise ValueError(f"x должен иметь 12 каналов, а получил {x.shape[1]}")

        x = self._normalize_like_flownet(x) # Нормализуем вход в соответствии с ожиданиями FlowNetS

        out_conv1 = self.flownets.conv1(x)
        out_conv2 = self.flownets.conv2(out_conv1)
        out_conv3 = self.flownets.conv3_1(self.flownets.conv3(out_conv2))
        out_conv4 = self.flownets.conv4_1(self.flownets.conv4(out_conv3))
        out_conv5 = self.flownets.conv5_1(self.flownets.conv5(out_conv4))
        out_conv6 = self.flownets.conv6_1(self.flownets.conv6(out_conv5))

        feat = self.pool(out_conv6) # (B, 1024, h, w) 
        feat = feat.flatten(start_dim=1) # (B, 1024, 1, 1) -> (B, 1024)
        feat = self.fc(feat) # (B, visual_dim)

        return feat

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
            input_size=self.VisualEncoder.out_dim+self.imuEncoder.out_dim,
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