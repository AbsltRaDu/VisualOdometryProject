import torch
import torch.nn as nn 

from src.geometry.Trajectory2DTorch import Trajectory2DTorch as TT2D

from src.function_of_loss.mse_pose import PoseLoss, PoseLossTrajectory, PoseLoseSeq, PoseLossTrajectorySeq

def angle_diff(pred_angle: torch.Tensor, fact_angle: torch.Tensor) -> torch.Tensor:
    '''
    Считает минимальную разницу между углами. Результат всегда лежит в диапазоне [-pi, pi]
    '''

    diff = pred_angle - fact_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    return diff

class PoseLoss2D(PoseLoss):
    
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        pred_p = y_pred[..., :2]
        pred_eul = y_pred[..., 2]
        
        fact_p = y_fact[..., :2]
        fact_eul = y_fact[..., 2]
        
        yaw_error = angle_diff(pred_eul, fact_eul)
        
        self.pos_loss = self.mse(pred_p, fact_p)
        self.r_loss = torch.mean(yaw_error ** 2)
        
        return self.pos_loss + self.k * self.r_loss


class PoseLossTrajectory2D(PoseLossTrajectory):
    
    def forward(self, y_pred: TT2D, y_fact: TT2D):
        
        r_pred = y_pred.poses.R
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.poses.R
        t_fact = y_fact.poses.t
        
        yaw_error = angle_diff(r_pred, r_fact)
        
        self.t_loss = self.mse(t_pred, t_fact)
        self.r_loss = torch.mean(yaw_error ** 2)
        
        return self.a * self.r_loss + self.b * self.t_loss
    
class PoseLoseSeq2D(PoseLoseSeq):
    
    def forward(self, y_pred: torch.Tensor, y_fact: torch.Tensor):
        
        if y_pred.ndim < 3 and y_fact.ndim < 3:
            raise ValueError('Последовательности должны иметь формат (N, seq, 6)')
        
        pred_p = y_pred[..., :2]
        pred_eul = y_pred[..., 2].unsqueeze(-1)
        
        fact_p = y_fact[..., :2]
        fact_eul = y_fact[..., 2].unsqueeze(-1)
        
        yaw_error = angle_diff(pred_eul, fact_eul)
        
        self.pos_loss = self.mse(pred_p, fact_p).sum(dim=(1, 2)).mean()
        self.r_loss = (yaw_error ** 2).sum(dim=(1, 2)).mean()
        
        return self.pos_loss + self.k * self.r_loss
    
class PoseLossTrajectorySeq2D(PoseLossTrajectorySeq):
    
    def forward(self, y_pred: TT2D, y_fact: TT2D):
        
        if not isinstance(y_pred, TT2D):
            raise TypeError('y_pred должен быть объектом TT2D')
        
        if not isinstance(y_fact, TT2D):
            raise TypeError('y_fact должен быть объектом PoseTorch')
        
        if y_pred.poses.t.ndim < 3 and y_fact.poses.t.ndim < 3:
            raise ValueError('Последовательности должны иметь формат (N, seq, 6)')
        
        
        r_pred = y_pred.poses.R.unsqueeze(-1)
        t_pred = y_pred.poses.t
        
        r_fact = y_fact.poses.R.unsqueeze(-1)
        t_fact = y_fact.poses.t
        
        yaw_error = angle_diff(r_pred, r_fact)
        
        # self.r_loss = self.mse(r_pred, r_fact).sum(dim=(1, 2)).mean()
        self.t_loss = self.mse(t_pred, t_fact).sum(dim=(1, 2)).mean()
        self.r_loss = (yaw_error ** 2).sum(dim=(1, 2)).mean()
        
        return self.a * self.r_loss + self.b * self.t_loss