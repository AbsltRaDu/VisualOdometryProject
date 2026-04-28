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
    
class PoseLoseSeq(PoseLoss):
    '''
    Ошибка покадровая по последовательности
    '''

    def __init__(self, k: float = 100, reduction='none'):
        super().__init__(k=k, reduction=reduction)
    
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        if y_pred.ndim < 3 and y_fact.ndim < 3:
            raise ValueError('Последовательности должны иметь формат (N, seq, 6)')
            
        pred_p = y_pred[..., :3]
        pred_eul = y_pred[..., 3:]
        
        fact_p = y_fact[..., :3]
        fact_eul = y_fact[..., 3:]
        
        self.pos_loss = self.mse(pred_p, fact_p).sum(dim=(1, 2)).mean()
        self.r_loss = self.mse(pred_eul, fact_eul).sum(dim=(1, 2)).mean()
        
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
    
class PoseLossTrajectorySeq(PoseLossTrajectory):
    '''
    Ошибка по конечной позе по последовательностям
    '''
    
    def __init__(self, a: float = 100, b: float = 1, reduction: str = 'none'):
        super().__init__(a=a, b=b, reduction=reduction)
        
    def forward(self, y_pred: TrajectoryTorch, y_fact: PoseTorch):
        
        if not isinstance(y_pred, TrajectoryTorch):
            raise TypeError('y_pred должен быть объектом TrajectoryTorch')
        
        if not isinstance(y_fact, PoseTorch):
            raise TypeError('y_fact должен быть объектом PoseTorch')
        
        if y_pred.poses.t.ndim < 3 and y_fact.t.ndim < 3:
            raise ValueError('Последовательности должны иметь формат (N, seq, 6)')
        
        
        r_pred = y_pred.poses.R.as_quat()
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.R.as_quat()
        t_fact = y_fact.t
        
        self.r_loss = self.mse(r_pred, r_fact).sum(dim=(1, 2)).mean()
        self.t_loss = self.mse(t_pred, t_fact).sum(dim=(1, 2)).mean()
        
        return self.a * self.r_loss + self.b * self.t_loss

     