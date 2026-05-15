import os

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import cv2
from tqdm import tqdm

import plotly.graph_objects as go

from src.dataloaders.datasets import mavDataset
from src.models.modelsClassic.LucaseKanade import LKOpticalFlowVO
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray



transform = T.Compose([
    # T.Resize((192, 320)),
    TorchImageToCvGray()
])

normalize = None

lst_of_dataset = os.listdir("datasets/simulation")
lst_of_dataset_test = lst_of_dataset[-1]

dataset = mavDataset(
    "datasets/simulation",
    transform=transform,
    normalize=normalize,
    device="cpu",
    lst_of_datasets=lst_of_dataset_test,
    # max_size=1000,
)

dtrain = data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
)

print("Длина датасета:", len(dataset))


model = LKOpticalFlowVO(
    width=752,          
    height=480,        
    new_width=752,      
    new_height=480,    
    fov_deg=90,
    baseline=0.2,

    # Параметры поиска точек
    max_corners=3000,
    quality_level=0.01,
    min_distance=7,
    block_size=7,

    # Параметры LK optical flow
    lk_win_size=(21, 21),
    lk_max_level=3,
    lk_max_error=30.0,
    fb_max_error=1.5,

    # Параметры глубины
    min_depth=0.2,
    max_depth=100.0,

    # Параметры PnP
    min_pnp_points=20,
    min_pnp_inliers=10,

    return_debug=True,
)

model.eval()


dtrain = iter(dtrain)

with torch.no_grad():
    img_left_1, img_right_1, img_left_2, img_right_2, y, pose = next(dtrain)

    print("Image shape:", img_left_1.shape)
    print("GT translation:")
    print(y[0, :3])

    img_left_1 = img_left_1[0].numpy().astype(np.uint8)
    img_right_1 = img_right_1[0].numpy().astype(np.uint8)
    img_left_2 = img_left_2[0].numpy().astype(np.uint8)


    y_pred, debug = model(img_left_1, img_right_1, img_left_2) # Оценка относительного движения через LK + StereoSGBM + PnP

    y_pred = y_pred.unsqueeze(0)

    # if not debug.get("success"):
    #     print("Первый шаг неуспешен:", debug)

    if normalize:
        y_pred = normalize.denormalize(y_pred)

    pose0 = PT.from_lie(pose)

    trajectory = TT.from_lie_relative(y_pred, pose0)
    trajectory_fact = TT.from_lie_relative(y, pose0)

    dataset_train = tqdm(dtrain, desc="Поехали", position=0)

    for item in dataset_train:
        img_left_1, img_right_1, img_left_2, img_right_2, y, _ = item

        print("Image shape:", img_left_1.shape)
        print("GT translation:")
        print(y[0, :3])
        
        img_left_1 = img_left_1[0].numpy().astype(np.uint8)
        img_right_1 = img_right_1[0].numpy().astype(np.uint8)
        img_left_2 = img_left_2[0].numpy().astype(np.uint8)

        y_pred, debug = model(img_left_1, img_right_1, img_left_2)

        y_pred = y_pred.unsqueeze(0)

        # if not debug.get("success"):
        #     print("Ошибка шага:", debug)

        if normalize:
            y_pred = normalize.denormalize(y_pred)

        # Накопление траекторий
        trajectory = trajectory.extend_lie_relative(y_pred)
        trajectory_fact = trajectory_fact.extend_lie_relative(y)

trajectory_fact = trajectory_fact.positions().squeeze().cpu()
trajectory = trajectory.positions().squeeze().cpu()

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory_fact],
    y=[x[1] for x in trajectory_fact],
    z=[x[2] for x in trajectory_fact],
    mode="lines",
    name="Фактическая траектория",
))

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory[:-1]],
    y=[x[1] for x in trajectory[:-1]],
    z=[x[2] for x in trajectory[:-1]],
    mode="lines",
    name="Предсказанная траектория",
))

fig.show()