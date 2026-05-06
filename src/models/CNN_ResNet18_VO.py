import torch
import torch.nn as nn 
import torchvision.models as models

class CNN_ResNet18_VO(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weight = models.ResNet18_Weights.IMAGENET1K_V1

        else: 
            weight = None
            
        resNet = models.resnet18(weights=weight)
        resNet.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 12 каналов, потому что у нас 4 изображения с двух кадров стерео и на каждом по 3 (RGB)
        resNet.fc = nn.Identity() # Выключаем FC, потому что пишем свой
        self.encoder = resNet # Бэкбон CNN 
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        p, r = self.fc1(x), self.fc2(x)
        return torch.hstack([p, r]).to(dtype=torch.float64)