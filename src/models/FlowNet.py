import torch 
import torch.nn as nn 

class conv(nn.Module):
    
    def __init__(self, in_c, out_c, kernel_size: int = 3, padding: int = 1, stride: int =1, batchNorm: bool = True, dropout: float = 0.3):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batchNorm
        self.dropout = dropout
        
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm, dropout=self.dropout, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        return self.layer(x)
    
class FlowNet(nn.Module):
    
    def __init__(self, in_c, out_c):

        self.layer1 = conv(in_c, 64, kernel_size=7, padding=3, stride=2, dropout=0.2)
        self.layer2 = conv(64, 128, kernel_size=5,  padding=2, stride=2, dropout=0.2)
        self.layer3 = conv(128, 256, kernel_size=5, padding=2, stride=2, dropout=0.2)
        self.layer4 = conv(256, 256, kernel_size=3, padding=1, stride=1, dropout=0.2)
        self.layer5 = conv(256, 512, kernel_size=3, padding=1, stride=2, dropout=0.2)
        self.layer6 = conv(512, 512, kernel_size=3, padding=1, stride=1, dropout=0.2)
        self.layer6 = conv(512, 512, kernel_size=3, padding=1, stride=2, dropout=0.2)
        self.layer6 = conv(512, 512, kernel_size=3, padding=1, stride=1, dropout=0.2)
        self.layer7 = conv(512, out_c, kernel_size=3, padding=1, stride=2, dropout=0.5)
        
        pass