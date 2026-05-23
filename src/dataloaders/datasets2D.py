

import torch

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.geometry.RotationTorch import RotationTorch as RT
from src.geometry.Pose2DTorch import Pose2DTorch as PT2D

class mavDatasetCNN_2D(mavDatasetCNN_3D):
    '''
    Модификация датасета для 2D пространства
    '''

    # Ф-ия вызова матрицы смещения и начальной позы
    def get_delta_quat(self, q1: torch.Tensor, q2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        
        yaw1 = RT.from_quat(q1).as_euler()[2]
        yaw2 = RT.from_quat(q2).as_euler()[2]
        xy1 = p1[:2]
        xy2 = p2[:2]
        
        pose1 = PT2D.from_xy_yaw(xy1, yaw1)
        pose2 = PT2D.from_xy_yaw(xy2, yaw2)
        
        delta_pose = pose1.inv() * pose2
        delta_pose = delta_pose.as_vector()
        
        if self.normalize is not None:
            delta_pose = self.normalize.normalize(delta_pose)
        
        return delta_pose, pose1.as_vector()
    
class SequenceDataset2D(mavDatasetCNN_2D):
    
    def __init__(self, dataset: torch.utils.data.Dataset, seq=10, stride=None):
        self.dataset = dataset
        self.seq = seq
        self.stride = stride if stride is not None else seq
        self.dataset_group = [(idxs[0], idxs[1] - self.seq + 1) for idxs in self.dataset.dataset_group] # Сохраняем инфу о конечном индексе для каждого датасета
    
        self.seq_starts = []

        for start, end in self.dataset.dataset_group:

            max_start = end - self.seq

            for idx in range(start, max_start + 1, self.stride):
                self.seq_starts.append(idx)

        self.dataset_group = [(0, len(self.seq_starts))]

    
    def __len__(self):
        return len(self.seq_starts)

        
    def __getitem__(self, item):
        start_idx = self.seq_starts[item]
        
        
        xs = []
        ys = []
        Ts = []
        
        for idx in range(start_idx, start_idx + self.seq):
            
            x, y, Tm = self.dataset[idx]
        
            xs.append(x)
            ys.append(y)
            Ts.append(Tm)
        
        x_seq = torch.stack(xs, dim=0)
        y_seq = torch.stack(ys, dim=0)
        T_seq = torch.stack(Ts, dim=0)

        return x_seq, y_seq, T_seq