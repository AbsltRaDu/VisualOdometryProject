import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.datasets_for_CNN_RAFT import mavDatasetCNN_RAFT, SequenceDataset_RAFT
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.CNN_RAFT import RAFTPoseCNN, RAFTPoseCNNEncoder
from src.models.DeepVO import DeepVO, DeepVO_RAFT
from src.function_of_loss.mse_pose import PoseLoseSeq, PoseLossTrajectorySeq
from src.piplines.pipline_learning_NN import training_RCNN_progressive_JointTrain_RAFT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')

transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(mean=[-1], std=[1])
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-1]


dataset_train = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test)

WINDOW_SIZE = 20

dataset_train = SequenceDataset_RAFT(dataset_train, seq=5)
dataset_test = SequenceDataset_RAFT(dataset_test, seq=5)

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

raft = RAFTPoseCNN()
raft = raft.to(device)

if os.path.isfile('process_of_fitting/fitting_models/CNN_RAFT.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNN_RAFT.tar', map_location=device)
    raft.load_state_dict(state_dict_cnn)
    print('Были загружены веса энкодера')

encoder = RAFTPoseCNNEncoder(raft=raft.flow_model, cnn=raft.CNN)
encoder = encoder.to(device)
model = DeepVO_RAFT(encoder=encoder, feat_dim=128, hidden_size=256)
model = model.to(device)

if os.path.isfile('process_of_fitting/fitting_models/RCNN_RAFT_test.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/RCNN_RAFT_test.tar', map_location=device)
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

# заморозили RAFT
for p in model.encoder.flow_model.parameters():
    p.requires_grad = False

epochs = 20
loss_func = PoseLoseSeq(k=1)
loss_func_trajectory = PoseLossTrajectorySeq()
optimizer = torch.optim.Adam([
    {
        "params": model.encoder.parameters(),
        "lr": 1e-6
    },
    {
        "params": model.head.rnn.parameters(),
        "lr": 1e-4
    },
    {
        "params": model.head.fc1.parameters(),
        "lr": 1e-4
    },
    {
        "params": model.head.fc2.parameters(),
        "lr": 1e-4
    }
])

count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_RCNN_progressive_JointTrain_RAFT(train_data, train_data, model, loss_func_pose=loss_func, loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, name_of_model=os.path.join('process_of_fitting/fitting_models', 'RCNN_RAFT_test.tar'), \
        path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'RCNN_RAFT_test.json'), squueze=False, window_size=WINDOW_SIZE, weight_trajectory=0.5)

