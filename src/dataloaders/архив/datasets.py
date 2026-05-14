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
from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
    
class mavDataset(mavDatasetCNN_3D):

    def __init__(self, path, transform=None, normalize=None, device='cpu',  lst_of_datasets=[], max_size=None):
        super().__init__(path=path, transform=transform, normalize=normalize, device=device, lst_of_datasets=lst_of_datasets, max_size=max_size)
        
    
    def get_mat_of_imgs(self, pair1_left, pair2_rigth):

        path_to_img_left_1, path_to_img_left_2 = pair1_left
        path_to_img_right_1, path_to_img4_right_2 = pair2_rigth
        
        img_left_1, img_left_2 = read_image(path_to_img_left_1), read_image(path_to_img_left_2)
        img_right_1, img_right_2 = read_image(path_to_img_right_1), read_image(path_to_img4_right_2)
        
        if img_left_1.shape[0] == 1:
            img_left_1 = img_left_1.repeat(3, 1, 1)
        if img_left_2.shape[0] == 1:
            img_left_2 = img_left_2.repeat(3, 1, 1)
        if img_right_1.shape[0] == 1:
            img_right_1 = img_right_1.repeat(3, 1, 1)
        if img_right_2.shape[0] == 1:
            img_right_2 = img_right_2.repeat(3, 1, 1)
        
        img_left_1 = img_left_1.float() / 255.0
        img_left_2 = img_left_2.float() / 255.0
        img_right_1 = img_right_1.float() / 255.0
        img_right_2 = img_right_2.float() / 255.0
        
        
        x = None
        if self.transform:
            img_left_1, img_left_2 = self.transform(img_left_1), self.transform(img_left_2)
            img_right_1, img_right_2 = self.transform(img_right_1), self.transform(img_right_2)
        
        return img_left_1, img_right_1, img_left_2, img_right_2
    
    def __getitem__(self, item):

        pair1 = self.lst_x1[item]
        pair2 = self.lst_x2[item]
        img_left_1, img_right_1, img_left_2, img_right_2 = self.get_mat_of_imgs(pair1, pair2)

        # Получаем координаты
        y, T_m = self.lst_target[item]
    
        return img_left_1, img_right_1, img_left_2, img_right_2, y, T_m
    
    
class SequenceDataset_classic(SequenceDataset):
    
    '''
    Обертка над mavDatasetCNN_3D для формирования последовательностей
    
    ВАЖНО: Не безопасно при прямом итерировании по датасету,
    так как датасет при getitem не учитывает границы датасетов.
    Границы учитываются в BatchSeqSampler и хранятся в параметре объекта класса seq
    '''
    
    
    def __init__(self, dataset: data.Dataset, seq=10, stride=None):
        super().__init__(dataset, seq, stride)
    
    def __getitem__(self, item):
        
        img1s = []
        img3s = []
        img2s = []
        img4s = []
        ys = []
        Ts = []
        
        for idx in range(item, item + self.seq):
            
            img1, img3, img2, img4, y, Tm = self.dataset[idx]
        
            img1s.append(img1)
            img3s.append(img3)
            img1s.append(img1)
            img4s.append(img4)
            ys.append(y)
            Ts.append(Tm)
        
        img1_seq = torch.stack(img1s, dim=0)
        img3_seq = torch.stack(img3s, dim=0)
        img2_seq = torch.stack(img2s, dim=0)
        img4_seq = torch.stack(img4s, dim=0)
        y_seq = torch.stack(ys, dim=0)
        T_seq = torch.stack(Ts, dim=0)

        return img1_seq, img3_seq, img2_seq, img4_seq, y_seq, T_seq