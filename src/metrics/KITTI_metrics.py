import torch

def translation_rmse_drift(p_pred, p_fact, path_lenght):
    drift = torch.linalg.norm(p_pred - p_fact, dim=1) 
    return torch.mean(drift / path_lenght) * 100

def rotation_rmse_drift(r_pred, r_fact, path_lenght):
    drift = torch.linalg.norm(r_pred - r_fact) 
    return torch.mean(drift / path_lenght) * 100

def rmse_pos(p_pred: torch.Tensor, p_fact: torch.Tensor) -> torch.Tensor:
    '''
    RMSE по координатам
    '''
    
    pred = p_pred[..., :3]
    fact  = p_fact[..., :3]
    
    