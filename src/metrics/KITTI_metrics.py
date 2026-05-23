import torch

from src.function_of_loss.mse_pose import RelativeMotionError

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

def get_KITTI_metrices(motion_fact, motion_pred, path_lenght):
    
    error = RelativeMotionError()(motion_pred, motion_fact)
    
    r_loss_abs = torch.linalg.norm(error.R.as_rotvec(), dim=-1)
    t_loss_abs = torch.linalg.norm(error.t, dim=-1)
    
    r_error = rotation_rmse_drift(r_loss_abs, path_lenght)
    t_error = translation_rmse_drift(t_loss_abs, path_lenght)
    
    return r_error, t_error

def get_KITTI_metrices2D(motion_fact, motion_pred, path_lenght):
    
    error = RelativeMotionError()(motion_pred, motion_fact)
    
    r_loss_abs = torch.linalg.norm(error.R, dim=-1)
    t_loss_abs = torch.linalg.norm(error.t, dim=-1)
    
    r_error = rotation_rmse_drift(r_loss_abs, path_lenght)
    t_error = translation_rmse_drift(t_loss_abs, path_lenght)
    
    return r_error, t_error