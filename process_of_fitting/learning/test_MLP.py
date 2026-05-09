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
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-1]


dataset_train = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test, max_size=30) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test, max_size=30)

WINDOW_SIZE = 1

dataset_train = SequenceDataset_RAFT(dataset_train, seq=5)
dataset_test = SequenceDataset_RAFT(dataset_test, seq=5)

train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=6, window_size=WINDOW_SIZE, shuffle=True)
test_sampler = ProgressiveWindowBatchSampler(dataset_test, batch_size=6, window_size=WINDOW_SIZE, shuffle=True)

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
model = DeepVO_RAFT(encoder=encoder, feat_dim=128, hidden_size=256, dropout=0)
model = model.to(device)

loss_func = PoseLoseSeq(k=1)
loss_func_trajectory = PoseLossTrajectorySeq()

img1, img2, y, pose = next(iter(train_data))

img1 = img1.to(device)
img2 = img2.to(device)
y = y.to(device).float()

model.eval()
with torch.no_grad():
    S = img1.shape[1] # берем длину последовательности
    features = []
    
    for s in range(S):
        img1_s = img1[:, s] # (B, C, H, W)
        img2_s = img2[:, s]
        feat_s = model.encoder(img1_s, img2_s)
        features.append(feat_s.flatten(1))
features = torch.stack(features, dim=1) # (B, S, F)    

print(features.shape, features.mean(), features.std())

head = torch.nn.Sequential(
    torch.nn.Linear(features.shape[-1], 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 6)
).to(device)

opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)

for name, p in model.named_parameters():
    if "rnn" in name or "fc" in name:
        print(name, p.requires_grad)

for i, group in enumerate(opt.param_groups):
    n_params = sum(p.numel() for p in group["params"])
    n_trainable = sum(p.numel() for p in group["params"] if p.requires_grad)
    print(i, "lr =", group["lr"], "params =", n_params, "trainable =", n_trainable)

model.train()
for i in range(1000):
    
    w_before = model.head.rnn.weight_ih_l0.detach().clone()
    opt.zero_grad()

    pred, _ = model(img1, img2, hidden=None)

    loss = loss_func(pred.float(), y.float())

    loss.backward()
    opt.step()
    
    
    w_after = model.head.rnn.weight_ih_l0.detach().clone()
    
    if i % 50 == 0:
        print(i, loss.item())
        print("RNN weight change:", (w_after - w_before).abs().max())
        print("grad norm:", model.head.rnn.weight_ih_l0.grad.norm())



