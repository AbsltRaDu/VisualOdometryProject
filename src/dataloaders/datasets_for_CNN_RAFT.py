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
from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
    
class mavDatasetCNN_RAFT(mavDatasetCNN_3D):

    def __init__(self, path, transform=None, normalize=None, device='cpu',  lst_of_datasets=[], max_size=None):
        super().__init__(path=path, transform=transform, normalize=normalize, device=device, lst_of_datasets=lst_of_datasets, max_size=max_size)
        
    
    def get_mat_of_imgs(self, pair1):

        path_to_img1, path_to_img2 = pair1
        
        # img1, img2 = Image.open(path_to_img1).convert('RGB'), Image.open(path_to_img2).convert('RGB')
        # img3, img4 = Image.open(path_to_img3).convert('RGB'), Image.open(path_to_img4).convert('RGB')
        
        img1, img2 = read_image(path_to_img1), read_image(path_to_img2)
        
        if img1.shape[0] == 1:
            img1 = img1.repeat(3, 1, 1)
        if img2.shape[0] == 1:
            img2 = img2.repeat(3, 1, 1)
        
        img1 = img1.float() / 255.0
        img2 = img2.float() / 255.0
        
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
            
        
        return img1, img2
    
    def __getitem__(self, item):

        pair1 = self.lst_x1[item]
        img1, img2 = self.get_mat_of_imgs(pair1)

        # Получаем координаты
        y, T_m = self.lst_target[item]
    
        return img1, img2, y, T_m
    
    
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