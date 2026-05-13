import os

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from tqdm import tqdm

import plotly.graph_objects as go

from src.dataloaders.datasetsINS import mavDatasetINS
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT
from src.models.modelsINS.PureINS import IMUPreprocessor, INSPropagator, PureINSModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Инференс на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Инференс на', device, sep=' ')

transform = None
normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
# lst_of_dataset_train = lst_of_dataset[-2]
lst_of_dataset_test = lst_of_dataset[-1]

dataset = mavDatasetINS('datasets/simulation', transform=transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test, num_of_imu=12) # Сразу формируем все массивы на GPU
dtrain = data.DataLoader(dataset=dataset, batch_size=1)

print('Длина датасета:', len(dataset), sep=' ')

preprocessor = IMUPreprocessor()
propagator = INSPropagator()
model = PureINSModel(preprocessor, propagator)

trajectory = []
trajectory_pred = []

b_g = torch.tensor([[0.0001, 0.0001, 0.0001]]).to(device)
b_a = b_g

dtrain = iter(dtrain)

model.eval()
with torch.no_grad():
    
    img_left_1, img_right_1, img_left_2, img_right_2, imu, imu_t, y, pose = next(dtrain)
    


    pose0 = pose.to(device)
    pose0 = PT.from_lie(pose0)
    imu = imu.to(device)
    imu_t = imu_t.to(device)
    init_velocity = torch.zeros(1, 3, device=device, dtype=torch.float64)
    y = y.to(device)
    
    y_pred, velocity, states = model(imu, imu_t, pose0, init_velocity, b_g=b_g, b_a=b_a)
    
    trajectory = TT.from_relative(y_pred, pose0)
    trajectory_fact = TT.from_lie_relative(y, pose0)
    
    

    dataset_train = tqdm(dtrain, desc=f'Поехали', position=0)
    for item in dataset_train:
        img_left_1, img_right_1, img_left_2, img_right_2, imu, imu_t, y, pose = item
        
        imu = imu.to(device)
        imu_t = imu_t.to(device)
        y = y.to(device)
        
        y_pred, velocity, states = model(imu, imu_t, trajectory.as_pose()[-1], velocity, b_g=b_g, b_a=b_a)
        
        dbg = model.propagator.debug
        
        # print('Траектория', trajectory.as_pose()[-1].t)
        # print("accel_body:", dbg["accel_body"][0])
        # print("f_world:", dbg["f_world"][0])
        # print("g:", dbg["g"][0])
        # print("a_world:", dbg["a_world"][0])
        # print("dt:", dbg["dt"][0])
        # print("velocity_new:", dbg["velocity_new"][0])
        
        trajectory = trajectory.extend_relative(y_pred)
        trajectory_fact = trajectory_fact.extend_lie_relative(y)

trajectory_fact = trajectory_fact.positions().squeeze().cpu()
trajectory = trajectory.positions().squeeze().cpu()

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