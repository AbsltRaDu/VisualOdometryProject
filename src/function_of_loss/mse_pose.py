import torch
import torch.nn as nn 

class PoseLoss(nn.Module):
    def __init__(self, k: float = 100, reduction: str = 'mean'):
        super().__init__()
        self.k = k # весовой коэф. для балансировки вкладов перемещения и вращения
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        pred_p = y_pred[:, :3]
        pred_eul = y_pred[:, 3:]
        
        fact_p = y_fact[:, :3]
        fact_eul = y_fact[:, 3:]
        
        self.pos_loss = self.mse(pred_p, fact_p)
        self.r_loss = self.mse(pred_eul, fact_eul)
        
        return self.pos_loss + self.k * self.r_loss
    
class PoseLossTrajectory(PoseLoss):
    
    def __init__(self, k: float = 100, reduction: str = 'mean'):
        super().__init__(self, k=k, reduction=reduction)
        
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        '''
        y_pred.size = (B, S, Dim)
        '''
        pass

     