import os

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import cv2
from tqdm import tqdm

import plotly.graph_objects as go

from src.dataloaders.datasets import mavDataset
from src.classic_models.FeaturesMethodModel import FeaturesMethod
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT
from src.classic_models.blocks.transforms_for_classic import TorchImageToCvGray

transform = T.Compose([
    # T.Resize((192, 320)),
    TorchImageToCvGray()
])

normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[-2]
lst_of_dataset_test = lst_of_dataset[-1]

dataset = mavDataset('datasets/simulation', transform=transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test) # Сразу формируем все массивы на GPU
dtrain = data.DataLoader(dataset=dataset, batch_size=1)

print('Длина датасета:', len(dataset), sep=' ')


# detect = cv2.SIFT_create(nfeatures=3000)
# matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

detect = cv2.ORB_create(nfeatures=3000)
matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)

model = FeaturesMethod(752, 480, 752, 480, 90, 0.12, detection_algoritm=detect, matcher=matcher, return_debug=True)

trajectory = []
trajectory_pred = []

# normalize = normalize.to(device)

dtrain = iter(dtrain)
model.eval()
with torch.no_grad():
    
    img1, img3, img2, img4, y, pose = next(dtrain)
    
    print(img1.shape)
    
    img1 = img1[0].numpy().astype(np.uint8)
    img2 = img2[0].numpy().astype(np.uint8)
    img3 = img3[0].numpy().astype(np.uint8)

    pose = pose
    y_pred, debug = model(img1, img3, img2)
    y_pred = y_pred.unsqueeze(0)
    y = y

    
    if not debug.get('success'):
        print(debug)
    
    if normalize:
        y_pred = normalize.denormalize(y_pred)
    
    pose0 = PT.from_lie(pose)
    trajectory = TT.from_lie_relative(y_pred, pose0)
    trajectory_fact = TT.from_lie_relative(y, pose0)
    
    dataset_train = tqdm(dtrain, desc=f'Поехали', position=0)
    for item in dataset_train:
        img1, img3, img2, img4, y, _ = item
        
        img1 = img1[0].numpy().astype(np.uint8)
        img2 = img2[0].numpy().astype(np.uint8)
        img3 = img3[0].numpy().astype(np.uint8)
        
        y_pred, debug = model(img1, img3, img2)
        y_pred = y_pred.unsqueeze(0)

        
        if not debug.get('success'):
            print(debug)
        
        if normalize:
            y_pred = normalize.denormalize(y_pred)
        
        trajectory = trajectory.extend_lie_relative(y_pred)
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