import os
import csv
import random

import torch
import torch.nn as nn 
import torchvision.models as models

from PIL import Image
from torchvision.io import read_image
from torch.utils import data
from torchvision import transforms as T

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.different_functions.RotationTorch import RotationTorch as RT

class mavDataLoader(data.Dataset):

    def __init__(self, path, transform=None, device='cpu', batchsize=8, hidden_size=0, lst_of_datasets=[]):
        self.transform = transform
        self.path = path
        self.device=device
        self.batch_groups = []
        self.dataset_group = []
        self.batchsize = batchsize
        self.hidden_size = hidden_size
        
        self.lst_target = [] # Словарь временных меток и значений по каждому датасету
        self.lst_x1 = [] # Путь до фото с камеры 1
        self.lst_x2 = [] # путь до фото сс камеры 2
        
        self.pair = [] # Пары изображений
        
        if not lst_of_datasets:
            lst_of_datasets = os.listdir(self.path)
        
        # Блок объединения датасетов
        self.start_idx = 0
        self.start_idx_dataset = 0
        for dataset in filter(lambda x: x in lst_of_datasets, os.listdir(self.path)):
            path_local = os.path.join(self.path, dataset)
        
        
            # Блок обработки координат
            with open(os.path.join(path_local, 'state_groundtruth_estimate0/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь таймстемпов с координатами
                targets = {int(dct['#timestamp']): [float(dct[' p_RS_R_x [m]']), float(dct[' p_RS_R_y [m]']), float(dct[' p_RS_R_z [m]']),
                                                    float(dct[' q_RS_w []']), float(dct[' q_RS_x []']), float(dct[' q_RS_y []']), float(dct[' q_RS_z []'])] for dct in csv.DictReader(f)}
                
            
            # Блок обработки камеры 0
            with open(os.path.join(path_local, 'cam0/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь камеры 1
                dct_of_timestamp_cam0 = {int(dct['#timestamp [ns]']): os.path.join(path_local, f"cam0/data/{dct['filename']}")  for dct in csv.DictReader(f)}
            
            # Блок обработки камеры 1
            with open(os.path.join(path_local, 'cam1/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь камеры 1
                dct_of_timestamp_cam1 = {int(dct['#timestamp [ns]']): os.path.join(path_local, f"cam1/data/{dct['filename']}") for dct in csv.DictReader(f)}

            keys = sorted(targets.keys())
            func_of_key = lambda x, y: x if x in y else min(y, key=lambda k: (abs(k - x), k))
            
            lst_target_coord = [targets[key] for key in keys]
            lst_cam0 = [dct_of_timestamp_cam0[func_of_key(key, dct_of_timestamp_cam0.keys())] for key in keys]
            lst_cam1 = [dct_of_timestamp_cam1[func_of_key(key, dct_of_timestamp_cam1.keys())] for key in keys]

            lst_target = []
            for coord1, coord2 in zip(lst_target_coord[:-1], lst_target_coord[1:]):
                p_coord1, q_coord1, p_coord2, q_coord2 = torch.tensor(coord1[:3], dtype=torch.float32), torch.tensor(coord1[3:], dtype=torch.float32), torch.tensor(coord2[:3], dtype=torch.float32), torch.tensor(coord2[3:], dtype=torch.float32)
                y, T_m = self.get_delta_quat(q_coord1, q_coord2, p_coord2, p_coord1)
                lst_target.append((y, T_m))
                
            lst_x1 = list(zip(lst_cam0[:-1], lst_cam0[1:]))
            lst_x2 = list(zip(lst_cam1[:-1], lst_cam1[1:]))
            
            if self.hidden_size:
                lst_target, lst_x1, lst_x2 = self.get_pairs(lst_target, lst_x1, lst_x2)
                
            self.batch_groups += self.get_batch_group(lst_target)     
            self.dataset_group.append(list(range(self.batch_groups[0][0], self.batch_groups[-1][-1])))

            
            self.lst_target += lst_target
            self.lst_x1 += lst_x1
            self.lst_x2 += lst_x2
            
        self.length = len(self.lst_target)

    def get_batch_group(self, lst_target):
        batch_groups = []
        for i in range(len(lst_target) // self.batchsize):
            end_idx = self.start_idx + self.batchsize
            batch_groups.append(list(range(self.start_idx, end_idx)))
            self.start_idx = end_idx
            
        return batch_groups
        
    def get_pairs(self, lst_target, lst_x1, lst_x2):
        lst_x1_new = []
        lst_x2_new = []
        lst_target_new = []
        for i in range(len(lst_target)-self.hidden_size):
            lst_x1_new.append(lst_x1[i:i+self.hidden_size])
            lst_x2_new.append(lst_x2[i:i+self.hidden_size])
            lst_target_new.append(lst_target[i+self.hidden_size-1])
        
        return lst_target_new, lst_x1_new, lst_x2_new
    
    def get_mat_of_imgs(self, pair1, pair2):

        path_to_img1, path_to_img2 = pair1
        path_to_img3, path_to_img4 = pair2
        # img1, img2 = Image.open(path_to_img1).convert('RGB'), Image.open(path_to_img2).convert('RGB')
        # img3, img4 = Image.open(path_to_img3).convert('RGB'), Image.open(path_to_img4).convert('RGB')
        
        img1, img2 = read_image(path_to_img1), read_image(path_to_img2)
        img3, img4 = read_image(path_to_img3), read_image(path_to_img4)
        
        if img1.shape[0] == 1:
            img1 = img1.repeat(3, 1, 1)
        if img2.shape[0] == 1:
            img2 = img2.repeat(3, 1, 1)
        if img3.shape[0] == 1:
            img3 = img3.repeat(3, 1, 1)
        if img4.shape[0] == 1:
            img4 = img4.repeat(3, 1, 1)
        
        img1 = img1.float() / 255.0
        img2 = img2.float() / 255.0
        img3 = img3.float() / 255.0
        img4 = img4.float() / 255.0
        
        
        x = None
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
            img3, img4 = self.transform(img3), self.transform(img4)
            x = torch.concat((img1, img3, img2, img4))
        
        return x
        
    def __getitem__(self, item):
        
        if self.hidden_size:
            lst_of_x = []
            for pair1, pair2 in zip(self.lst_x1[item], self.lst_x2[item]):
                lst_of_x.append(self.get_mat_of_imgs(pair1, pair2))
                
            x = torch.stack(lst_of_x)
        
        else:
            pair1, pair2 = self.lst_x1[item], self.lst_x2[item]
            x = self.get_mat_of_imgs(pair1, pair2)
    
        
        
        # Получаем координаты
        y, T_m = self.lst_target[item]
    
        return x, y, T_m
    
    def get_motion_matrix(self, R_matrix, delt_p):
        R_t = torch.cat((R_matrix, delt_p.unsqueeze(0).T), dim=1)
        T_m = torch.cat([R_t, torch.tensor([[0, 0, 0, 1]])], dim=0)

        return T_m
    
    # Ф-ия 
    def get_delta_quat(self, q1: torch.Tensor, q2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        q1_wxyz = q1.numpy() 
        q2_wxyz = q2.numpy()
        
        q1_xyzw = q1_wxyz[[1, 2, 3, 0]]
        q2_xyzw = q2_wxyz[[1, 2, 3, 0]]
        
        r1 = R.from_quat(q1_xyzw) # созадние объекта ориентации
        r2 = R.from_quat(q2_xyzw)
        
        r1m = torch.tensor(r1.as_matrix(), dtype=torch.float32) # Создание ориентации 1 кадра
        T1 = self.get_motion_matrix(r1m, p1)
        delt_p = r1m.T @ (p1-p2)
        
        dr = r2 * r1.inv()
        dq_matrix = dr.as_matrix() # Выводим поворот уже в виде матрицы, а не кватериона
        angles = dr.as_euler('xyz', degrees=False) # Вытаскиваем углы Эйлера в радианах
        
        return torch.cat((delt_p, torch.tensor(angles, dtype=torch.float32)), dim=0), T1
    
    def __len__(self):
        return self.length
    
    
class SequenceBatchSampler(data.Sampler):
    def __init__(self, batch_groups, shuffle=False):
        self.batch_groups = batch_groups
        self.shuffle = shuffle

    def __iter__(self):
        groups = self.batch_groups.copy()
        if self.shuffle:
            random.shuffle(groups)
        for group in groups:
            yield group

    def __len__(self):
        return len(self.batch_groups)
    
class mavDatasetCNN_3D(mavDataLoader):
    '''
    Датасет для CNN но с геометрией из PyTorch3D на самописной RotationTorch - аналоге Rotation из SciPy для работы на CUDA
    '''
    
    def __init__(self, path, transform=None, device='cpu', batchsize=8, hidden_size=0, lst_of_datasets=[]):
        super().__init__(path, transform=transform, device=device, batchsize=batchsize, hidden_size=hidden_size, lst_of_datasets=lst_of_datasets)
        
    def get_delta_quat(self, q1, q2, p1, p2):
        r1 = RT.from_quat(q1)
        r2 = RT.from_quat(q2)
        
        T1 = self.get_motion_matrix(r1, p1)
        delt_p = r1.inv().apply(p1 - p2)
        
        dr = r2 * r1.inv()
        angles = dr.as_euler()
        
        return torch.cat((delt_p, angles), dim=0), T1

    
