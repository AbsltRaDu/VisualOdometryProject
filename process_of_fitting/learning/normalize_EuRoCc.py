
import os
import torch
from torch.utils import data
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.normalize.PoseNormolizerLie import PoseNormalizerLie

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

lst_of_datasets = os.listdir('datasets/simulation')

dataset = mavDatasetCNN_3D('datasets/simulation', transform, device='cpu', lst_of_datasets=lst_of_datasets[:-1]) # Сразу формируем все массивы на GPU

mass = torch.stack([i[0] for i in dataset.lst_target])

print('Длина датасета:', len(dataset), sep=' ')
print('Длина датасета для вычисления параметров стандартизации', len(mass), sep=' ')

normalize = PoseNormalizerLie()
normalize.fit(mass)

normalize.save('process_of_fitting/normalize_params')