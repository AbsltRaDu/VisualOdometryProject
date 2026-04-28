import torch
import torch.nn as nn 

from src.geometry.PoseTorch import PoseTorch
from src.geometry.TrajectoryTorch import TrajectoryTorch

class PoseLoss(nn.Module):
    '''
    Ошибка по смещнию покадровому
    '''
    
    def __init__(self, k: float = 100, reduction: str = 'mean'):
        super().__init__()
        self.k = k # весовой коэф. для балансировки вкладов перемещения и вращения
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        pred_p = y_pred[..., :3]
        pred_eul = y_pred[..., 3:]
        
        fact_p = y_fact[..., :3]
        fact_eul = y_fact[..., 3:]
        
        self.pos_loss = self.mse(pred_p, fact_p)
        self.r_loss = self.mse(pred_eul, fact_eul)
        
        return self.pos_loss + self.k * self.r_loss
    
class PoseLossTrajectory(nn.Module):
    '''
    Ошибка по конечной позе
    '''
    
    def __init__(self, a: float = 100, b: float = 1, reduction: str = 'sum'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.a = a
        self.b = b
        
    def forward(self, y_pred: TrajectoryTorch, y_fact: PoseTorch):
        
        r_pred = y_pred.poses.R.as_quat()
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.R.as_quat()
        t_fact = y_fact.t
        
        self.r_loss = self.mse(r_pred, r_fact)
        self.t_loss = self.mse(t_pred, t_fact)
        
        return self.a * self.r_loss + self.b * self.t_loss

     