import os
import csv
import random
import bisect

import torch
import torch.nn as nn 
import torchvision.models as models

from PIL import Image
from torchvision.io import read_image
from torch.utils import data
from torchvision import transforms as T

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.geometry.RotationTorch import RotationTorch as RT
from src.geometry.PoseTorch import PoseTorch as PT

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D

class kittiDataset_3D(mavDatasetCNN_3D):
    
    def __init__(self, path, path_gt, transform=None, normalize=None, device='cpu', lst_of_datasets=[], stereo: bool = False, max_size=None):
        self.transform = transform
        self.path = path
        self.path_gt = path_gt
        self.device=device
        self.dataset_group = []
        self.max_size = max_size
        self.stereo = stereo
        
        self.normalize = normalize
        
        self.lst_target = [] # Словарь временных меток и значений по каждому датасету
        self.lst_x1 = [] # Путь до фото с камеры 1
        self.lst_x2 = [] # путь до фото сс камеры 2
        
        self.pair = [] # Пары изображений
        
        if not lst_of_datasets:
            lst_of_datasets = os.listdir(self.path)
        
        # Блок объединения датасетов
        self._get_data(lst_of_datasets)
        
        if self.max_size is not None:
            self.lst_target = self.lst_target[:self.max_size]
            self.lst_x1 = self.lst_x1[:self.max_size]
            self.lst_x2 = self.lst_x2[:self.max_size]

            self.dataset_group = [(0, len(self.lst_target))]
        
        self.length = len(self.lst_target)
    

    def _get_data_from_path(self, path_local):
        """
        path_local: datasets/KITTI/sequences/00
        root_poses: datasets/KITTI/poses
        """

        seq_name = os.path.basename(path_local)  


        pose_file = os.path.join(self.path_gt, f"{seq_name}.txt")

        targets = {}

        with open(pose_file, 'r') as f:
            for idx, line in enumerate(f):
                values = list(map(float, line.strip().split()))

                
                T = torch.tensor(values, dtype=torch.float64).view(3, 4)
                bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float64)
                T = torch.cat([T, bottom], dim=0)

                targets[idx] = T  


        img_dir_0 = os.path.join(path_local, 'image_2')

        files_0 = sorted([f for f in os.listdir(img_dir_0) if f.endswith('.png')])

        dct_of_timestamp_cam0 = {
            idx: os.path.join(img_dir_0, fname)
            for idx, fname in enumerate(files_0)
        }

        img_dir_1 = os.path.join(path_local, 'image_3')

        files_1 = sorted([f for f in os.listdir(img_dir_1) if f.endswith('.png')])

        dct_of_timestamp_cam1 = {
            idx: os.path.join(img_dir_1, fname)
            for idx, fname in enumerate(files_1)
        }
        
        
        # Блок синхронизации длинны 
        n = min(len(targets), len(dct_of_timestamp_cam0), len(dct_of_timestamp_cam1))
        targets = {k: targets[k] for k in range(n)}
        dct_of_timestamp_cam0 = {k: dct_of_timestamp_cam0[k] for k in range(n)}
        dct_of_timestamp_cam1 = {k: dct_of_timestamp_cam1[k] for k in range(n)}

        return targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1

    # Ф-ия вызова матрицы смещения и начальной позы
    def get_delta_quat(self, pose1: torch.Tensor, pose2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        
        pose1 = PT.from_matrix(pose1)
        pose2 = PT.from_matrix(pose2)
        
        delta_pose = pose1.inv() * pose2
        delta_pose = delta_pose.as_lie()
        
        if self.normalize is not None:
            delta_pose = self.normalize.normalize(delta_pose)
        
        return delta_pose, pose1.as_lie()    # Ф-ия вызова матрицы смещения и начальной позы
    
    def _get_coord(self, targets, keys):
        lst_target_coord = [targets[key] for key in keys]
        lst_target = []
        for pose1, pose2 in zip(lst_target_coord[:-1], lst_target_coord[1:]):
            
            y, T_m = self.get_delta_quat(pose1, pose2)
            
            lst_target.append((y, T_m))
            
        return lst_target    

    def _get_data(self, lst_of_datasets):
         
        # Блок объединения датасетов
        self.start_idx = 0
        self.start_idx_dataset = 0
        
        for dataset in filter(lambda x: x in lst_of_datasets, os.listdir(self.path)):
            path_local = os.path.join(self.path, dataset)
        
            targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1 = self._get_data_from_path(path_local)
            
            keys = sorted(targets.keys())
            
            lst_cam0 = [dct_of_timestamp_cam0[key] for key in keys] # в KITTI Уже все синхронизированно
            lst_cam1 = [dct_of_timestamp_cam1[key] for key in keys]

            lst_target = self._get_coord(targets, keys)
            
            self._get_group_of_dataset(lst_target) # Записали границы датасета
            
            lst_x1 = list(zip(lst_cam0[:-1], lst_cam0[1:]))
            lst_x2 = list(zip(lst_cam1[:-1], lst_cam1[1:]))
            
            self.lst_target += lst_target
            self.lst_x1 += lst_x1
            self.lst_x2 += lst_x2


class kittiDataset_3D_euler(kittiDataset_3D):
    
        # Ф-ия вызова матрицы смещения и начальной позы
    def get_delta_quat(self, pose1: torch.Tensor, pose2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        
        pose1 = PT.from_matrix(pose1)
        pose2 = PT.from_matrix(pose2)
        
        delta_pose = pose1.inv() * pose2
        delta_pose = delta_pose.as_euler()
        
        if self.normalize is not None:
            delta_pose = self.normalize.normalize(delta_pose)
        
        return delta_pose, pose1.as_euler()    # Ф-ия вызова матрицы смещения и начальной позы