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

from src.geometry.RotationTorch import RotationTorch as RT
from src.geometry.PoseTorch import PoseTorch as PT
    
class mavDatasetCNN_3D(data.Dataset):

    def __init__(self, path, transform=None, normalize=None, device='cpu',  lst_of_datasets=[], max_size=None):
        self.transform = transform
        self.path = path
        self.device=device
        self.dataset_group = []
        self.max_size = max_size
        
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
    
    @staticmethod
    def _get_data_from_path(path_local):
        
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

        return targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1

    def _get_coord(self, targets, keys):
        lst_target_coord = [targets[key] for key in keys]
        lst_target = []
        for coord1, coord2 in zip(lst_target_coord[:-1], lst_target_coord[1:]):
            p_coord1, q_coord1 = torch.tensor(coord1[:3], dtype=torch.float64), torch.tensor(coord1[3:], dtype=torch.float64)
            p_coord2, q_coord2 = torch.tensor(coord2[:3], dtype=torch.float64), torch.tensor(coord2[3:], dtype=torch.float64)
            y, T_m = self.get_delta_quat(q_coord1, q_coord2, p_coord1, p_coord2)
            lst_target.append((y, T_m))
            
        return lst_target
    
    def _get_group_of_dataset(self, lst_target):
        '''
        Ф-ия для сохранения информации о начальных и конечных индексах датасетов
        '''
        
        end_indx = self.start_idx_dataset + len(lst_target)
        self.dataset_group.append((self.start_idx_dataset, end_indx))
        self.start_idx_dataset = end_indx
        
    def _get_data(self, lst_of_datasets):
         
        # Блок объединения датасетов
        self.start_idx = 0
        self.start_idx_dataset = 0
        
        for dataset in filter(lambda x: x in lst_of_datasets, os.listdir(self.path)):
            path_local = os.path.join(self.path, dataset)
        
            targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1 = self._get_data_from_path(path_local)
            
            keys = sorted(targets.keys())
            func_of_key = lambda x, y: x if x in y else min(y, key=lambda k: (abs(k - x), k))
            
            lst_cam0 = [dct_of_timestamp_cam0[func_of_key(key, dct_of_timestamp_cam0.keys())] for key in keys]
            lst_cam1 = [dct_of_timestamp_cam1[func_of_key(key, dct_of_timestamp_cam1.keys())] for key in keys]

            lst_target = self._get_coord(targets, keys)
            
            self._get_group_of_dataset(lst_target) # Записали границы датасета
            
            lst_x1 = list(zip(lst_cam0[:-1], lst_cam0[1:]))
            lst_x2 = list(zip(lst_cam1[:-1], lst_cam1[1:]))
            
            self.lst_target += lst_target
            self.lst_x1 += lst_x1
            self.lst_x2 += lst_x2
    
    
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
    
    # Ф-ия вызова матрицы смещения и начальной позы
    def get_delta_quat(self, q1: torch.Tensor, q2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        
        pose1 = PT.from_rt(RT.from_quat(q1), p1)
        pose2 = PT.from_rt(RT.from_quat(q2), p2)
        
        delta_pose = pose1.inv() * pose2
        delta_pose = delta_pose.as_lie()
        
        if self.normalize is not None:
            delta_pose = self.normalize.normalize(delta_pose)
        
        return delta_pose, pose1.as_lie()
    
    def __getitem__(self, item):

        pair1, pair2 = self.lst_x1[item], self.lst_x2[item]
        x = self.get_mat_of_imgs(pair1, pair2)

        # Получаем координаты
        y, T_m = self.lst_target[item]
    
        return x, y, T_m
    
    def __len__(self):
        return self.length
    
class SequenceDataset(data.Dataset):
    
    '''
    Обертка над mavDatasetCNN_3D для формирования последовательностей
    
    ВАЖНО: Не безопасно при прямом итерировании по датасету,
    так как датасет при getitem не учитывает границы датасетов.
    Границы учитываются в BatchSeqSampler и хранятся в параметре объекта класса seq
    '''
    
    
    def __init__(self, dataset: data.Dataset, seq=10):
        self.dataset = dataset
        self.seq = seq
        self.dataset_group = [(idxs[0], idxs[1] - self.seq + 1) for idxs in self.dataset.dataset_group] # Сохраняем инфу о конечном индексе для каждого датасета
    
    def get_dataset_id(self, item):
        for dataset_id, _tpl in enumerate(self.dataset_group):
            if _tpl[0] <= item <= _tpl[1]:
                return dataset_id

        raise IndexError(f'Индекс {item} вне границ dataset_group')
    
    def __len__(self):
        return len(self.dataset) - self.seq

        
    def __getitem__(self, item):
        
        xs = []
        ys = []
        Ts = []
        
        for idx in range(item, item + self.seq):
            
            x, y, Tm = self.dataset[idx]
        
            xs.append(x)
            ys.append(y)
            Ts.append(Tm)
        
        x_seq = torch.stack(xs, dim=0)
        y_seq = torch.stack(ys, dim=0)
        T_seq = torch.stack(Ts, dim=0)

        return x_seq, y_seq, T_seq