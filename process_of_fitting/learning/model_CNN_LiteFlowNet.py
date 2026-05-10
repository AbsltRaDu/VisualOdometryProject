import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from dataloaders.datasets import mavDatasetCNN_RAFT
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.CNN_LiteFlowNet import LiteFlowNetPoseCNN
from src.function_of_loss.mse_pose import PoseLoss, PoseLossTrajectory
from src.piplines.pipline_learning_NN import training_RAFT_progressive

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


dataset_train = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test)

WINDOW_SIZE = 10

train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=32, window_size=WINDOW_SIZE, shuffle=True)
test_sampler = ProgressiveWindowBatchSampler(dataset_test, batch_size=32, window_size=WINDOW_SIZE, shuffle=True)

train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True)
test_data = data.DataLoader(dataset_test, batch_sampler=test_sampler, num_workers=6, pin_memory=True)

print('Длина тренировочного выборки:', len(dataset_train), sep=' ')
print('Длина тестовой выборки:', len(dataset_test), sep=' ')

example_of_obj = next(iter(train_data))
print('Размерность IMG1:', example_of_obj[0].shape, sep=' ')
print('Размерность IMG2:', example_of_obj[1].shape, sep=' ')
print('Размерность y:', example_of_obj[2].shape, sep=' ')
print('Размерность T_m:', example_of_obj[3].shape, sep=' ')

model = LiteFlowNetPoseCNN()
model = model.to(device)

if os.path.isfile('process_of_fitting/fitting_models/CNN_LiteFlowNet.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNN_LiteFlowNet.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

# заморозили LiteFlowNet
for p in model.encoder.liteflownet.parameters():
    p.requires_grad = True

epochs = 20
loss_func = PoseLoss(k=1)
loss_func_trajectory = PoseLossTrajectory(reduction='mean')
optimizer = torch.optim.Adam([
    {
        "params": model.encoder.liteflownet.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.encoder.CNN.parameters(),
        "lr": 1e-4
    },
    {
        "params": model.fc1.parameters(),
        "lr": 1e-4
    },
    {
        "params": model.fc2.parameters(),
        "lr": 1e-4
    }
])

count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_RAFT_progressive(train_data, test_data, model, loss_func_pose=loss_func, loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, name_of_model=os.path.join('process_of_fitting/fitting_models', 'CNN_LiteFlowNet.tar'), \
        path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'CNN_LiteFlowNet.json'), squueze=False, window_size=WINDOW_SIZE)

