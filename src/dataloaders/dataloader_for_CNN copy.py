import os
import csv
import random

import torch
import torch.nn as nn 
import torchvision.models as models

from PIL import Image
from torch.utils import data, Sampler
from torchvision import transforms as T

import numpy as np
from scipy.spatial.transform import Rotation as R

class mavDataLoader(data.Dataset):

    def __init__(self, path, transform=None, device='cpu', batchsize=8):
        self.transform = transform
        self.path = path
        self.device=device
        self.batchsize = batchsize
        
        self.dct_of_timestamp_with_coord = {} # Словарь временных меток и значений по каждому датасету
        self.dct_of_timestamp_cam0 = {} # Путь до фото с камеры 1
        self.dct_of_timestamp_cam1 = {} # путь до фото сс камеры 2
        self.pair = [] # Пары изображений
         
        # Блок объединения датасетов
        for dataset in filter(lambda x: x in ['mav0_easy1'], os.listdir(self.path)):
            path_local = os.path.join(self.path, dataset)
        
        
            # Блок обработки координат
            with open(os.path.join(path_local, 'state_groundtruth_estimate0/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь таймстемпов с координатами
                self.dct_of_timestamp_with_coord.update({(path_local, int(dct['#timestamp'])): [float(dct[' p_RS_R_x [m]']), float(dct[' p_RS_R_y [m]']), float(dct[' p_RS_R_z [m]']),
                                                    float(dct[' q_RS_w []']), float(dct[' q_RS_x []']), float(dct[' q_RS_y []']), float(dct[' q_RS_z []'])] for dct in csv.DictReader(f)})
        
            # Блок обработки камеры 0
            with open(os.path.join(path_local, 'cam0/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь камеры 1
                self.dct_of_timestamp_cam0[path_local] = {int(dct['#timestamp [ns]']): dct['filename'] for dct in csv.DictReader(f)}
            
            # Блок обработки камеры 1
            with open(os.path.join(path_local, 'cam1/data.csv'), 'r', encoding='utf-8') as f: # Вытаскиваю словарь камеры 1
                self.dct_of_timestamp_cam1[path_local] = {int(dct['#timestamp [ns]']): dct['filename'] for dct in csv.DictReader(f)}
        
                # Создаем последовательные во времени пары кадров
                frames = sorted(self.dct_of_timestamp_with_coord.keys(), key=lambda x: x[1])
                pair = list(zip(frames[:-1], frames[1:]))
                
                batch_end = 0

                for _ in range(len(pair) // batchsize):
                    batch_start = batch_end
                    batch_end = batch_end + batchsize
                    self.pair.append(pair[batch_start:batch_end])
        
        
        self.length = len(self.pair)
        
    def __getitem__(self, item):
        
        X = []
        Y = []
        T_M = []
        
        batch = self.pair[item]
        
        for pair in batch:

            pic1, pic2 = pair
            path_1, timestamp_1 = pic1
            path_2, timestamp_2 = pic2
            
            # Получаем изображения с первой камеры
            func_of_key = lambda x, y: x if x in y else min(y, key=lambda k: (abs(k - x), k))
            key1, key2 = func_of_key(timestamp_1, self.dct_of_timestamp_cam0[path_1].keys()), func_of_key(timestamp_2, self.dct_of_timestamp_cam0[path_2].keys())
            key3, key4 = func_of_key(timestamp_1, self.dct_of_timestamp_cam1[path_1].keys()), func_of_key(timestamp_2, self.dct_of_timestamp_cam1[path_2].keys())
            path_to_img1, path_to_img2 = os.path.join(path_1, f"cam0/data/{self.dct_of_timestamp_cam0[path_1][key1]}"), os.path.join(path_2, f"cam0/data/{self.dct_of_timestamp_cam0[path_2][key2]}")
            path_to_img3, path_to_img4 = os.path.join(path_1, f"cam1/data/{self.dct_of_timestamp_cam1[path_1][key1]}"), os.path.join(path_2, f"cam1/data/{self.dct_of_timestamp_cam1[path_2][key2]}")
            img1, img2 = Image.open(path_to_img1).convert('RGB'), Image.open(path_to_img2).convert('RGB')
            img3, img4 = Image.open(path_to_img3).convert('RGB'), Image.open(path_to_img4).convert('RGB')
            
            x = None
            if self.transform:
                img1, img2 = self.transform(img1), self.transform(img2)
                img3, img4 = self.transform(img3), self.transform(img4)
                x = torch.concat((img1, img3, img2, img4)).to(self.device)
            X.append(x)
            
            # Получаем координаты
            coord1, coord2 = self.dct_of_timestamp_with_coord[pic1], self.dct_of_timestamp_with_coord[pic2]
            p_coord1, q_coord1, p_coord2, q_coord2 = torch.tensor(coord1[:3], dtype=torch.float32, device=self.device), torch.tensor(coord1[3:], dtype=torch.float32, device=self.device), torch.tensor(coord2[:3], dtype=torch.float32, device=self.device), torch.tensor(coord2[3:], dtype=torch.float32, device=self.device)
            y, T_m = self.get_delta_quat(q_coord1, q_coord2, p_coord2, p_coord1)
            Y.append(y)
            T_M.append(T_m)
        
        # return X, Y, T_M
        return torch.stack(X), torch.stack(Y), torch.stack(T_M)
    
    def get_motion_matrix(self, R_matrix, delt_p):
        R_t = torch.cat((R_matrix, delt_p.unsqueeze(0).T), dim=1)
        T_m = torch.cat([R_t, torch.tensor([[0, 0, 0, 1]], device=self.device)], dim=0)

        return T_m
    
    # Ф-ия 
    def get_delta_quat(self, q1: torch.Tensor, q2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor): # Ф-ия поиска определения углой эйлера из кватерионов + определение прирощения позы
        q1_wxyz = q1.cpu().numpy() 
        q2_wxyz = q2.cpu().numpy()
        
        q1_xyzw = q1_wxyz[[1, 2, 3, 0]]
        q2_xyzw = q2_wxyz[[1, 2, 3, 0]]
        
        r1 = R.from_quat(q1_xyzw) # созадние объекта ориентации
        r2 = R.from_quat(q2_xyzw)
        
        r1m = torch.tensor(r1.as_matrix(), dtype=torch.float32).to(self.device) # Создание ориентации 1 кадра
        T1 = self.get_motion_matrix(r1m, p1)
        delt_p = r1m.T @ (p1-p2)
        
        dr = r2 * r1.inv()
        dq_matrix = dr.as_matrix() # Выводим поворот уже в виде матрицы, а не кватериона
        angles = dr.as_euler('xyz', degrees=False) # Вытаскиваем углы Эйлера в радианах
        
        return torch.cat((delt_p, torch.tensor(angles, device=self.device, dtype=torch.float32)), dim=0), T1
    
    def __len__(self):
        return self.length
    
    
    class SequenceBatchSampler(Sampler):
        def __init__(self, batch_groups, shuffle=True):
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