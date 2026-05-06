import torch
from torch.utils import data
from torchvision import transforms as T
import os

from tqdm import tqdm

import plotly.graph_objects as go

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.models.CNN_ResNet18_VO import CNN_ResNet18_VO
from src.models.DeepVO import DeepVO, PairwiseVOModel, DeepVORNN
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Инференс на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Инференс на', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[-2]
# lst_of_dataset_test = lst_of_dataset[-1]

dataset = mavDatasetCNN_3D('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_train) # Сразу формируем все массивы на GPU
dtrain = data.DataLoader(dataset=dataset, batch_size=1)

print('Длина датасета:', len(dataset), sep=' ')

model_cnn = CNN_ResNet18_VO()
state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet18_VO_SW_test.tar', map_location=device)
model_cnn.load_state_dict(state_dict_cnn)
model_cnn = model_cnn.to(device)

trajectory = []
trajectory_pred = []

normalize = normalize.to(device)

dtrain = iter(dtrain)
model_cnn.eval()
with torch.no_grad():
    
    x, y, pose = next(dtrain)
    pose = pose.to(device)
    y_pred = model_cnn(x.to(device))
    y = y.to(device)
    
    if normalize:
        y_pred = normalize.denormalize(y_pred)
    
    pose0 = PT.from_lie(pose)
    trajectory = TT.from_lie_relative(y_pred, pose0)
    trajectory_fact = TT.from_lie_relative(y, pose0)
    
    dataset_train = tqdm(dtrain, desc=f'Поехали', position=0)
    for item in dataset_train:
        x, y, _ = item
        
        y_pred = model_cnn(x.to(device))
        
        if normalize:
            y_pred = normalize.denormalize(y_pred)
        
        trajectory = trajectory.extend_lie_relative(y_pred)
        trajectory_fact = trajectory_fact.extend_lie_relative(y)
        
trajectory_fact = trajectory_fact.positions().squeeze(0).cpu()
trajectory = trajectory.positions().squeeze(0).cpu()

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory_fact],
    y=[x[1] for x in trajectory_fact],
    z=[x[2] for x in trajectory_fact],
    mode='lines',
    name='Фактическая траектория'
))

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory[:-1]],
    y=[x[1] for x in trajectory[:-1]],
    z=[x[2] for x in trajectory[:-1]],
    mode='lines',
    name='Предсказанная траектория'
))

fig.show()