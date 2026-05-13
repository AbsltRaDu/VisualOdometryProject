import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.datasets_for_CNN import mavDatasetVIO, SequenceDatasetIMU
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.modelsNN.ViNet import VisualEncoder, imuEncoder, ViNet
from src.function_of_loss.mse_pose import PoseLoseSeq, PoseLossTrajectorySeq
from src.piplines.pipline import trainingProgressiveVIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')

transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-1]


dataset_train = mavDatasetVIO('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetVIO('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test)

WINDOW_SIZE = 20

dataset_train = SequenceDatasetIMU(dataset_train, seq=5)
dataset_test = SequenceDatasetIMU(dataset_test, seq=5)

train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=32, window_size=WINDOW_SIZE, shuffle=True)
test_sampler = ProgressiveWindowBatchSampler(dataset_test, batch_size=32, window_size=WINDOW_SIZE, shuffle=True)

train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_data = data.DataLoader(dataset_test, batch_sampler=test_sampler, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2)

print('Длина тренировочного выборки:', len(dataset_train), sep=' ')
print('Длина тестовой выборки:', len(dataset_test), sep=' ')

example_of_obj = next(iter(train_data))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность IMU:', example_of_obj[1].shape, sep=' ')
print('Размерность y:', example_of_obj[3].shape, sep=' ')
print('Размерность T_m:', example_of_obj[4].shape, sep=' ')

visual_encoder = VisualEncoder(checkpoint_path='/home/aleksandr/Desktop/learning/flownet2-pytorch/checkpoints/FlowNet2-S_checkpoint.pth.tar')
visual_encoder = visual_encoder.to(device)

imu_encoder = imuEncoder()
imu_encoder = imu_encoder.to(device)

model = ViNet(VisualEncoder=visual_encoder, imuEncoder=imu_encoder, dropout=0.1)
model = model.to(device)

if os.path.isfile('process_of_fitting/fitting_models/ViNet_test.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/ViNet_test.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

# # заморозили RAFT
# for p in model.VisualEncoder.liteflownet.parameters():
#     p.requires_grad = False

epochs = 20
loss_func = PoseLoseSeq(k=1)
loss_func_trajectory = PoseLossTrajectorySeq()
optimizer = torch.optim.Adam([
    {
        "params": model.VisualEncoder.fc.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.imuEncoder.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.lstm.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.fc1.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.fc2.parameters(),
        "lr": 1e-6
    }
])

count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

pipline = trainingProgressiveVIO(train_data=train_data, test_data=train_data, model=model, loss_func_pose=loss_func, 
                       loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, name_of_model=os.path.join('process_of_fitting/fitting_models', 'ViNet_test.tar'), \
        path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'ViNet_test.json'), window_size=WINDOW_SIZE, weight_trajectory=0.5)

dct_of_results = pipline()   
