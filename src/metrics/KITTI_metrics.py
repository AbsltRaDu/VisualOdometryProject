import torch

def translation_rmse_drift(p_lose, path_lenght):
    '''
    Метрика KITTI Odometry MRSE Pose Lose к длине пути
    '''
    
    return torch.mean(p_lose / path_lenght) * 100

def rotation_rmse_drift(r_lose, path_lenght):
    '''
    Метрика KITTI Odometry RMSE Rotation Lose к длине пути
    '''
    
    return torch.mean(r_lose / path_lenght) * 100

    
    