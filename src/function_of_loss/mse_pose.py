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

class PoseLossEuler(PoseLoss):
    
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        pred_p = y_pred[..., 3:].flatten()
        pred_eul = y_pred[..., :3].flatten()
        
        fact_p = y_fact[..., 3:].flatten()
        fact_eul = y_fact[..., :3].flatten()
        
        print("y_pred.shape:", y_pred.shape)
        print("y_fact.shape:", y_fact.shape)
        print("y_pred dtype:", y_pred.dtype)
        print("y_fact dtype:", y_fact.dtype)
        
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
        
    def forward(self, y_pred: TrajectoryTorch, y_fact: TrajectoryTorch):
        
        r_pred = y_pred.poses.R.as_quat()
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.poses.R.as_quat()
        t_fact = y_fact.poses.t
        
        self.r_loss = self.mse(r_pred, r_fact)
        self.t_loss = self.mse(t_pred, t_fact)
        
        return self.a * self.r_loss + self.b * self.t_loss
    

    
class PoseLossTrajectorySeq(PoseLossTrajectory):
    '''
    Ошибка по конечной позе по последовательностям
    '''
    
    def __init__(self, a: float = 100, b: float = 1, reduction: str = 'none'):
        super().__init__(a=a, b=b, reduction=reduction)
        
    def forward(self, y_pred: TrajectoryTorch, y_fact: TrajectoryTorch):
        
        if not isinstance(y_pred, TrajectoryTorch):
            raise TypeError('y_pred должен быть объектом TrajectoryTorch')
        
        if not isinstance(y_fact, TrajectoryTorch):
            raise TypeError('y_fact должен быть объектом PoseTorch')
        
        if y_pred.poses.t.ndim < 3 and y_fact.poses.t.ndim < 3:
            raise ValueError('Последовательности должны иметь формат (N, seq, 6)')
        
        
        r_pred = y_pred.poses.R.as_quat()
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.poses.R.as_quat()
        t_fact = y_fact.poses.t
        
        self.r_loss = self.mse(r_pred, r_fact).sum(dim=(1, 2)).mean()
        self.t_loss = self.mse(t_pred, t_fact).sum(dim=(1, 2)).mean()
        
        return self.a * self.r_loss + self.b * self.t_loss

class RelativeMotionError(nn.Module):
    
    def __init__(self, a: float = 100, b: float = 1):
        super().__init__()
        
        self.a = a
        self.b = b
        
    def forward(self, y_pred: PoseTorch, y_fact: PoseTorch):
        
        motion_error = y_fact.inv() * y_pred

        return motion_error
        
        

class WeightedMSELoss(nn.Module):
    """
    Computes a weighted MSE loss for angle and translation components.

    Args:
        window_size (int): Used to reshape tensors for weighted loss calculation.
        alpha (float, optional): Weight for the angle loss component.
    """

    def __init__(
        self,
        window_size: int=2,
        alpha: float=1,
    ):
        super(WeightedMSELoss, self).__init__()
        self.window_size = window_size
        self.alpha = float(alpha)
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Separate angles and translation for ground truth
        y_true = torch.reshape(y_true, (y_true.shape[0], self.window_size - 1, 6))
        gt_angles = y_true[:, :, :3].flatten()
        gt_translation = y_true[:, :, 3:].flatten()

        # Separate angles and translation for predicted
        y_pred = torch.reshape(y_pred, (y_pred.shape[0], self.window_size - 1, 6))
        estimated_angles = y_pred[:, :, :3].flatten()
        estimated_translation = y_pred[:, :, 3:].flatten()

        # Calculate weighted losses for angles and translation
        loss_angles = self.mse_loss(estimated_angles, gt_angles)
        loss_translation = self.mse_loss(estimated_translation, gt_translation)

        # Compute final weighted loss
        loss = loss_translation + self.alpha * loss_angles

        return loss