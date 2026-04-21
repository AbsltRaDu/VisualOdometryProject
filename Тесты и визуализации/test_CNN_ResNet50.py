import torch
from torchvision import transforms as T

from tqdm import tqdm

import plotly.graph_objects as go

from src.dataloaders.dataloader_for_CNN import mavDataLoader
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.models.DeepVO import DeepVO, PairwiseVOModel, DeepVORNN
from src.geometry.trigan import euler_to_matrix_R, get_motion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])
dataset_train = mavDataLoader('datasets/euroc_mav', transform=transform, lst_of_datasets=['mav0_vic2'])

print('Длина датасета:', len(dataset_train), sep=' ')

model_cnn = CNN_ResNet50_VO()
state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet50_VO_2_0.tar')
model_cnn.load_state_dict(state_dict_cnn)
model_cnn = model_cnn.to(device)

trajectory = []
trajectory_pred = []

dtrain = iter(dataset_train)
model_cnn.eval()
with torch.no_grad():
    
    x, y, t_m = next(dtrain)
    y_pred = model_cnn(x.unsqueeze(0).to(device)).squeeze().cpu()
    r_pred = euler_to_matrix_R(y_pred[3:])
    motion_pred = get_motion_matrix(r_pred, y_pred[:3])
    
    r = euler_to_matrix_R(y[3:])
    motion = get_motion_matrix(r, y[:3])
    
    trajectory.append(t_m @ motion)
    trajectory_pred.append(t_m @ motion_pred)
    
    dataset_train = tqdm(dtrain, desc=f'Поехали', position=0)
    for item in dataset_train:
        x, y, _ = item
        
        y_pred = model_cnn(x.unsqueeze(0).to(device)).squeeze().cpu()
        r_pred = euler_to_matrix_R(y_pred[3:])
        motion_pred = get_motion_matrix(r_pred, y_pred[:3])
        
        r = euler_to_matrix_R(y[3:])
        motion = get_motion_matrix(r, y[:3])

        T = trajectory[-1]
        T_pred = trajectory_pred[-1]
        
        trajectory.append(T @ motion)
        trajectory_pred.append(T_pred @ motion_pred)
    
        
trajectory_3d = [tr[:3, 3] for tr in trajectory]
trajectory_pred_3d = [tr[:3, 3] for tr in trajectory_pred]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory_3d],
    y=[x[1] for x in trajectory_3d],
    z=[x[2] for x in trajectory_3d],
    mode='lines',
    name='Фактическая траектория'
))

fig.add_trace(go.Scatter3d(
    x=[x[0] for x in trajectory_pred_3d],
    y=[x[1] for x in trajectory_pred_3d],
    z=[x[2] for x in trajectory_pred_3d],
    mode='lines',
    name='Предсказанная траектория'
))

fig.show()