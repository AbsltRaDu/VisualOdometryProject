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

class LiteFlowNetMatching(nn.Module):
    
    def __init__(self, level=-1):
        super().__init__()
        
        self.encoder = FlowNetS()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        '''
        6-ти канал
        '''
        
        
        x = self.encoder(x)
        p, r = self.fc1(x), self.fc2(x)
        
        return torch.hstack([p, r]).to(dtype=torch.float64)
        
    