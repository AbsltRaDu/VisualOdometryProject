import torch
import torch.nn as nn

from models_from_github.liteflownet.run import Network
from models_from_github.liteflownet.run import correlation

class LiteFlowNetMatching(nn.Module):
    
    def __init__(self, level=-1):
        super().__init__()
        
        torch.set_grad_enabled(True)
        liteflownet = Network()
        
        self.features = liteflownet.netFeatures
        self.correlation_fn = correlation.FunctionCorrelation
        self.level = level

    def forward(self, img1, img2):
        feats1, feats2 = self.features(img1), self.features(img2)
        feats1, feats2 = feats1[self.level], feats2[self.level]
        
        corr = self.correlation_fn(
            tenOne=feats1,
            tenTwo=feats2,
            intStride=1
        )
        
        return corr

class LiteFlowNetEncoder(nn.Module):
    def __init__(self, level, corr_channels):
        super().__init__()
        
        self.liteflownet = LiteFlowNetMatching(level=level)
        self.CNN = nn.Sequential(
            nn.Conv2d(corr_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def forward(self, img1, img2):
        corr = self.liteflownet(img1, img2)
        x = self.CNN(corr)
        x = x.flatten(1)
        
        return x

class LiteFlowNetPoseCNN(nn.Module):
    def __init__(self, level=-1, corr_channels=49):
        super().__init__()

        self.encoder = LiteFlowNetEncoder(level, corr_channels)

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
        x = self.encoder(img1, img2)
        p, r = self.fc1(x), self.fc2(x)

        
        return torch.hstack([p, r]).to(dtype=torch.float64)
    