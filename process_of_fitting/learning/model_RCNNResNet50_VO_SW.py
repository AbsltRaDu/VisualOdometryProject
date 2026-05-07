import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.models.DeepVO_CNN import DeepVO_CNN
from src.function_of_loss.mse_pose import PoseLoseSeq, PoseLossTrajectorySeq
from src.piplines.pipline_learning_NN import training_RCNN_progressive_JointTrain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')

WINDOW_SIZE = 10

# lst_of_datasets_for_train = ['mav0']
# lst_of_datasets_for_test = ['mav0']

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-1]

# lst_of_dataset_train = ['maw_straight_line', 'maw_forward_backward', 'maw_up_down', 'maw_yaw_only', 'maw_turning_motion', 
#                         'maw_turns_yaw_snake_400m', 'maw_turns_yaw_left_right_300m', 'maw_altitude_steps_350m', 'maw_altitude_wave_500m', 'maw_mixed_random_short_500m',
#                         'maw_hover_stop_and_go_300m', 'maw_hover_yaw_hover_250m']
# lst_of_dataset_test = ['maw_square']

# print(f'Для валидации используется датасет: {lst_of_dataset_test}')

dataset_train = mavDatasetCNN_3D('datasets/simulation', transform, normalize=None, device='cpu', lst_of_datasets=lst_of_dataset_train) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetCNN_3D('datasets/simulation', transform, normalize=None, device='cpu', lst_of_datasets=lst_of_dataset_test)

dataset_train = SequenceDataset(dataset_train, seq=WINDOW_SIZE)
dataset_test = SequenceDataset(dataset_test, seq=WINDOW_SIZE)

train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=16, window_size=WINDOW_SIZE, shuffle=True)
test_sampler = ProgressiveWindowBatchSampler(dataset_test, batch_size=16, window_size=WINDOW_SIZE, shuffle=False)

train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True)
test_data = data.DataLoader(dataset_test, batch_sampler=test_sampler, num_workers=6, pin_memory=True)

print('Длина тренировочного выборки:', len(dataset_train), sep=' ')
print('Длина тестовой выборки:', len(dataset_test), sep=' ')

example_of_obj = next(iter(train_data))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
print('Размерность T_m:', example_of_obj[2].shape, sep=' ')

# encoder = CNN_ResNet50_VO()
# state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet50_VO_SW.tar', map_location=device)
# encoder.load_state_dict(state_dict_cnn)
# print('Веса энкодера были загружены')

model = DeepVO_CNN()
model = model.to(device)

if os.path.isfile('process_of_fitting/fitting_models/RCNN_VO_SW.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/RCNN_VO_SW.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

# # всё заморозили
# for p in model.parameters():
#     p.requires_grad = False

# # обучаем новый первый слой
# for p in model.encoder.conv1.parameters():
#     p.requires_grad = True

# # обучаем самый верхний блок resNet
# for p in model.encoder.layer4.parameters():
#     p.requires_grad = True

# # обучаем полносвязки
# for p in model.fc1.parameters():
#     p.requires_grad = True

# for p in model.fc2.parameters():
#     p.requires_grad = True

epochs = 50
loss_func = PoseLoseSeq(k=1)
loss_func_trajectory = PoseLossTrajectorySeq()

optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.rnn.parameters(), "lr": 1e-4},
    {"params": model.fc1.parameters(), "lr": 1e-4},
    {"params": model.fc2.parameters(), "lr": 1e-4},
], weight_decay=1e-4)


count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_RCNN_progressive_JointTrain(train_data, train_data, model, loss_func_pose=loss_func, loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, name_of_model=os.path.join('process_of_fitting/fitting_models', 'RCNN_VO_SW_test.tar'), \
        path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'RCNN_VO_SW_test.json'), squueze=False, window_size=WINDOW_SIZE, weight_trajectory=0.05)

