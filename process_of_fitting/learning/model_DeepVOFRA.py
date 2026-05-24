import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
from src.dataloaders.datasets_KITTI import kittiDataset_3D
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
# from src.models.modelsNN.DeepVO import VisualEncoder, DeepVO
from src.models.modelsNN.DeepVOFRA import DeepVO
from src.function_of_loss.mse_pose import PoseLoseSeq, PoseLossTrajectorySeq
from src.piplines.pipline import TrainerRNN, TrainerDeepVO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')

transform = T.Compose([
    T.Resize((192, 640)),
    T.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    )
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

# lst_of_dataset = [i for i in os.listdir('datasets/sequences') if i != 'poses']
# lst_of_dataset_train = ["00", "02", "08", "09"]
# lst_of_dataset_test = ["03"]

# dataset_train = kittiDataset_3D(
#     'datasets/sequences',
#     'datasets/sequences/poses',
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_train,
#     stereo=False
#     )

# dataset_test = kittiDataset_3D(
#     'datasets/sequences',
#     'datasets/sequences/poses',
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=False
#     )

lst_of_dataset = os.listdir('datasets/simulation_2')
lst_of_dataset_test = ['mav_look_forward_rectangle_500m', 
                       'mav_look_forward_climb_soft_turns_450m', 
                       'mav_look_forward_soft_s_curve_500m', 
                       'mav_look_forward_long_soft_zigzag_altitude_750m',
                       'mav_look_forward_descent_soft_turns_450m']
lst_of_dataset_train = [i for i in lst_of_dataset if i not in lst_of_dataset_test]
lst_of_dataset_test = ['mav_look_forward_square_400m']

# lst_of_dataset_test = ['mav_mixed_random_short_500m']


dataset_train = mavDatasetCNN_3D(
    'datasets/simulation_2', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_train
    ) 

dataset_test = mavDatasetCNN_3D(
    'datasets/simulation_2', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test
    )

WINDOW_SIZE = 10

dataset_train = SequenceDataset(dataset_train, seq=10)
dataset_test = SequenceDataset(dataset_test, seq=10)

train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=8, window_size=WINDOW_SIZE, shuffle=True)
test_sampler = ProgressiveWindowBatchSampler(dataset_test, batch_size=8, window_size=WINDOW_SIZE, shuffle=False)

train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_data = data.DataLoader(dataset_test, batch_sampler=test_sampler, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2)

print('Длина тренировочного выборки:', len(dataset_train), sep=' ')
print('Длина тестовой выборки:', len(dataset_test), sep=' ')

example_of_obj = next(iter(train_data))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
print('Размерность T_m:', example_of_obj[2].shape, sep=' ')


model = DeepVO()
model = model.to(device)

# path = "process_of_fitting/fitting_models/checkpoint_e190.pth"
# checkpoint = torch.load(path, map_location="cpu", weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])

if os.path.isfile('process_of_fitting/fitting_models/DeepVO_AirSim.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/DeepVO_AirSim.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

for param in model.feature_extractor.parameters():
    param.requires_grad = False

for param in model.feature_extractor.conv1.parameters():
    param.requires_grad = True

epochs = 50
# loss_func = PoseLossEuler(k=100)
loss_func = PoseLoseSeq()
loss_func_trajectory = PoseLossTrajectorySeq()
optimizer = torch.optim.Adam([
    {
        "params": model.feature_extractor.conv1.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.lstm.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.fc.parameters(),
        "lr": 1e-6
    }
],
weight_decay=1e-4)

count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

pipline = TrainerDeepVO(
    train_data=train_data,
    test_data=test_data,
    model=model,
    loss_func_pose=loss_func,
    loss_func_trajectory=loss_func_trajectory,
    optimizer=optimizer,
    epochs=epochs,
    device=device,
    normalize=normalize,
    name_of_model=os.path.join('process_of_fitting/fitting_models', 'DeepVO_AirSim.tar'),
    path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'DeepVO_AirSim.json'),
    window_size=WINDOW_SIZE,
    weight_trajectory=0
)

dct_of_results = pipline()   
