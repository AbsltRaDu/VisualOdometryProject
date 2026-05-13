import torch
from torch.utils import data
from torchvision import transforms as T
import os
from torchvision.models.optical_flow import raft_small

from src.dataloaders.datasetsINS import mavDatasetINS
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie


# from models_from_github.liteflownet.run import Network



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-3]

dataset_train = mavDatasetINS('datasets/simulation', transform=transform, lst_of_datasets=lst_of_dataset_test, num_of_imu=20)

print('Длина датасета:', len(dataset_train), sep=' ')

img_left_1, img_right_1, img_left_2, img_right_2, imu, imu_t, y, T_m = dataset_train[0]
img_left_1_1, img_right_1_1, img_left_2_1, img_right_2_1, imu_1, imu_t_1, y_1, T_m_1 = dataset_train[0]

imu = torch.stack([imu, imu_1])
imu_t = torch.stack([imu_t, imu_t_1])

for d in dataset_train:
    img_left_1, img_right_1, img_left_2, img_right_2, imu, imu_t, y, T_m = d
    print(imu.shape)

    




