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
from src.dataloaders.datasets import mavDataset


class mavDatasetINS(mavDataset):
    
    def __init__(self, path, transform=None, normalize=None, device='cpu',  lst_of_datasets=[], max_size=None, num_of_imu=10):
        '''
        Расширение датасета с добавление данных IMU для визуально-инерциальной одометрии
        
        '''
        
        self.transform = transform
        self.path = path
        self.device=device
        self.dataset_group = []
        self.max_size = max_size
        self.num_of_imu = num_of_imu
        
        self.normalize = normalize
        
        self.lst_target = [] # Словарь временных меток и значений по каждому датасету
        self.lst_x1 = [] # Путь до фото с камеры 1
        self.lst_x2 = [] # путь до фото сс камеры 2
        self.lst_imu = []
        self.lst_t = []
        
        self.pair = [] # Пары изображений
        
        if not lst_of_datasets:
            lst_of_datasets = os.listdir(self.path)
        
        # Блок объединения датасетов
        self._get_data(lst_of_datasets)
        
        if self.max_size is not None:
            self.lst_target = self.lst_target[:self.max_size]
            self.lst_x1 = self.lst_x1[:self.max_size]
            self.lst_x2 = self.lst_x2[:self.max_size]
            self.lst_imu = self.lst_imu[:self.max_size]

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

        with open(os.path.join(path_local, 'imu0/data.csv'), 'r', encoding='utf-8') as f:
            dct_of_timestamp_imu = {int(dct['#timestamp [ns]']): [float(dct['w_RS_S_x [rad s^-1]']), float(dct['w_RS_S_y [rad s^-1]']), float(dct['w_RS_S_z [rad s^-1]']),
                                                                  float(dct['a_RS_S_x [m s^-2]']), float(dct['a_RS_S_y [m s^-2]']), float(dct['a_RS_S_z [m s^-2]'])]
                                    for dct in csv.DictReader(f)}
         
        
        return targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1, dct_of_timestamp_imu
    
    @staticmethod
    def __get_imu_slices(lst_x1, imu_keys_sorted, dct_of_timestamp_imu, k=10):
        lst_imu = []
        lst_dt = []
        
        for t0, t1 in lst_x1:
            
            imu_slice = []
            imu_t = []
            mark = True
            
            i0 = bisect.bisect_left(imu_keys_sorted, t0)
            i1 = bisect.bisect_left(imu_keys_sorted, t1)
            
            imu_t = imu_keys_sorted[i0:i1]
            
            if len(imu_t) < 2:
                imu_t = [t0, t1]    
                mark = False
            
            
            imu_t = imu_t[-(k+1):]
            dt = [(_time1 - _time0) * 1e-9 for _time0, _time1 in zip(imu_t[:-1], imu_t[1:])] if len(imu_t) >= 2  else [(t1 - t0) * 1e-9]
                
            if mark:
                imu_slice = [dct_of_timestamp_imu[t] for t in imu_t[1:]]    
                
            if len(imu_slice) == 0:
                imu_slice.append([0, 0, 0, 0, 0, 0])
                
                
            if len(imu_slice) < k:
                pad_size = k - len(imu_slice)
            
                pad_value = imu_slice[0]
                pad_value_t = dt[0]
            
                pad = [pad_value.copy() for _ in range(pad_size)]
                pad_t = [pad_value_t for _ in range(pad_size)]
                imu_slice = pad + imu_slice
                dt = pad_t + dt
                
            dt = torch.tensor(dt, dtype=torch.float64)
            imu_tensor = torch.tensor(imu_slice, dtype=torch.float64)
            lst_imu.append(imu_tensor)
            lst_dt.append(dt.unsqueeze(-1))
            
        return lst_imu, lst_dt
    
    def _get_data(self, lst_of_datasets):
         
        # Блок объединения датасетов
        self.start_idx = 0
        self.start_idx_dataset = 0
        
        for dataset in filter(lambda x: x in lst_of_datasets, os.listdir(self.path)):
            path_local = os.path.join(self.path, dataset)
        
            targets, dct_of_timestamp_cam0, dct_of_timestamp_cam1, dct_of_timestamp_imu = self._get_data_from_path(path_local)
            
            keys = sorted(targets.keys())
            func_of_key = lambda x, y: x if x in y else min(y, key=lambda k: (abs(k - x), k))
            
            # Блок синхронизации камер и GT
            lst_cam0_ts = [func_of_key(key, dct_of_timestamp_cam0.keys()) for key in keys]
            lst_cam1_ts = [func_of_key(key, dct_of_timestamp_cam1.keys()) for key in keys]
            
            lst_cam0 = [dct_of_timestamp_cam0[key] for key in lst_cam0_ts]
            lst_cam1 = [dct_of_timestamp_cam1[key] for key in lst_cam1_ts]            

            lst_target = self._get_coord(targets, keys)
            
            self._get_group_of_dataset(lst_target) # Записали границы датасета
            
            lst_x1_ts = list(zip(lst_cam0_ts[:-1], lst_cam0_ts[1:]))
            lst_x1 = list(zip(lst_cam0[:-1], lst_cam0[1:]))
            lst_x2 = list(zip(lst_cam1[:-1], lst_cam1[1:]))
            
            # Блок синхронизации INS и камер
            imu_keys_sorted = sorted(dct_of_timestamp_imu.keys())
            lst_imu, lst_t = self.__get_imu_slices(lst_x1_ts, imu_keys_sorted, dct_of_timestamp_imu, k=self.num_of_imu) 
            
            self.lst_target += lst_target
            self.lst_x1 += lst_x1
            self.lst_x2 += lst_x2
            self.lst_imu += lst_imu
            self.lst_t += lst_t

    def __getitem__(self, item):

        pair1 = self.lst_x1[item]
        pair2 = self.lst_x2[item]
        img_left_1, img_right_1, img_left_2, img_right_2 = self.get_mat_of_imgs(pair1, pair2)
        
        # Вытаскиваем инерциалку
        imu = self.lst_imu[item]
        imu_t = self.lst_t[item]

        # Получаем координаты
        y, T_m = self.lst_target[item]
    
        return img_left_1, img_right_1, img_left_2, img_right_2, imu, imu_t, y, T_m