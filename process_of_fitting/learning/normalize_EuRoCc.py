

import torch
from torch.utils import data
from torchvision import transforms as T

from dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.normalize.PoseNormolizerLie import PoseNormalizerLie

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

lst_of_datasets = ['mav0_easy1', 'mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']
lst_of_datasets_for_tests = ['mav0_easy1']

dataset = mavDatasetCNN_3D('datasets/euroc_mav', transform, device='cpu', batchsize=16, lst_of_datasets=lst_of_datasets_for_tests) # Сразу формируем все массивы на GPU

groups = dataset.batch_groups.copy()

train_size = int(0.8 * len(groups))
indx = [i for lst in groups[:train_size] for i in lst]
mass = torch.stack([i[0] for i in dataset.lst_target])[indx]

print('Длина датасета:', len(dataset), sep=' ')
print('Длина датасета для вычисления параметров стандартизации', len(mass), sep=' ')

normalize = PoseNormalizerLie()
normalize.fit(mass)

normalize.save('process_of_fitting/normalize_params')