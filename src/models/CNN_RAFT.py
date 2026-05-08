import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class RAFTPoseCNN(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Raft_Small_Weights.DEFAULT
        self.flow_model = raft_small(weights=weights)

        

        self.CNN = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

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

    def forward(self, img1, img2):

        flows = self.flow_model(img1, img2)
        flow = flows[-1]  # (B, 2, H, W)

        x = self.CNN(flow)
        x = x.flatten(1)
        p, r = self.fc1(x), self.fc2(x)
        return torch.hstack([p, r]).to(dtype=torch.float64)
    
class RAFTPoseCNNEncoder(nn.Module):
    
    def __init__(self, raft, cnn):
        super().__init__()
        
        self.flow_model = raft
        self.CNN = cnn
        
    def forward(self, img1, img2):
        
        flows = self.flow_model(img1, img2)
        flow = flows[-1]  # (B, 2, H, W)

        x = self.CNN(flow)
        
        return x