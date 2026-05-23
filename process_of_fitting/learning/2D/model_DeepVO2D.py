import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T

from src.dataloaders.datasets2D import mavDatasetCNN_2D, SequenceDataset2D
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.modelsNN.DeepVO import VisualEncoder
from src.models.modelsNN.model2D.DeepVO2D import DeepVO2D
from src.function_of_loss.mse_pose2D import PoseLoseSeq2D, PoseLossTrajectorySeq2D
from src.piplines.pipline import TrainerRNN2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')

transform = T.Compose([
    T.Resize((448, 256)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[-4:-1]
lst_of_dataset_test = lst_of_dataset[-1]
# lst_of_dataset_test = ['mav_mixed_random_short_500m']


dataset_train = mavDatasetCNN_2D(
    'datasets/simulation', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_train
    ) 

dataset_test = mavDatasetCNN_2D(
    'datasets/simulation', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test
    )

WINDOW_SIZE = 10

dataset_train = SequenceDataset2D(dataset_train, seq=10)
dataset_test = SequenceDataset2D(dataset_test, seq=10)

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

encoder = VisualEncoder(checkpoint_path='/home/aleksandr/learning/diplom/flownet2-pytorch/checkpoints/FlowNet2-S_checkpoint.pth.tar')
encoder = encoder.to(device)
model = DeepVO2D(VisualEncoder=encoder)
model = model.to(device)

if os.path.isfile('process_of_fitting/fitting_models/DeepVO2D.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/DeepVO2D.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

epochs = 50
loss_func = PoseLoseSeq2D(k=100)
loss_func_trajectory = PoseLossTrajectorySeq2D()
optimizer = torch.optim.Adam([
    {
        "params": model.VisualEncoder.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.rnn.parameters(),
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

pipline = TrainerRNN2D(
    train_data=train_data,
    test_data=test_data,
    model=model,
    loss_func_pose=loss_func,
    loss_func_trajectory=loss_func_trajectory,
    optimizer=optimizer,
    epochs=epochs,
    device=device,
    normalize=normalize,
    name_of_model=os.path.join('process_of_fitting/fitting_models', 'DeepVO2D.tar'),
    path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'DeepVO2D.json'),
    window_size=WINDOW_SIZE,
    weight_trajectory=0
)

dct_of_results = pipline()   
