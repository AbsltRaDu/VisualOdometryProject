import torch

def euler_to_matrix_R(tns, device='cpu'): # Преобразованеи углов эйлера в матрицу вращения
    roll, pitch, yaw = tns # крен, тангаж, рысканье
    
    Qx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ], device=device)

    Qy = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ], device=device)
    
    Qz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ], device=device)
    
    return Qz @ Qy @ Qx

def get_motion_matrix(R_matrix, delt_p, device='cpu'):
    R_t = torch.cat((R_matrix, delt_p.unsqueeze(0).T), dim=1)
    T = torch.cat([R_t, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)

    return T

def R_mat_to_euler_and_pose(R_matrix, device='cpu'):
    roll = torch.atan2(R_matrix[2, 1], R_matrix[2, 2])
    pitch = torch.asin(-R_matrix[2, 0])
    yaw = torch.atan2(R_matrix[1, 0], R_matrix[0, 0])

    x, y, z = R_matrix[0, 3], R_matrix[1, 3], R_matrix[2, 3]
    return torch.tensor([roll, pitch, yaw, x, y, z], device=device)