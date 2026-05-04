import torch
from torch.utils import data
from torchvision import transforms as T

from tqdm import tqdm

import plotly.graph_objects as go

from dataloaders.datasets_for_CNN import mavDataLoader, mavDatasetCNN_3D, SequenceBatchSampler
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.models.DeepVO import DeepVO, PairwiseVOModel, DeepVORNN
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


lst_of_datasets = ['mav0_easy1', 'mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']
lst_of_datasets_for_tests = ['mav0_easy1']

dataset = mavDatasetCNN_3D('datasets/euroc_mav', transform, normalize=None, device='cpu', batchsize=16, lst_of_datasets=lst_of_datasets_for_tests) # Сразу формируем все массивы на GPU

groups = dataset.batch_groups.copy()

# Блок урезания для теста пайплайна
# len_groups = int(0.01 * len(groups))
# groups = groups[:len_groups]

train_size = int(0.8 * len(groups))
train_groups = groups[:train_size]
train_sampler = SequenceBatchSampler(train_groups)
train_data = data.DataLoader(dataset, batch_sampler=train_sampler, num_workers=6, pin_memory=True)

print('Длина датасета:', len(train_groups), sep=' ')

model_cnn = CNN_ResNet50_VO()
state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet50_VO.tar')
model_cnn.load_state_dict(state_dict_cnn)
model_cnn = model_cnn.to(device)

trajectory = []
trajectory_pred = []

dtrain = iter(train_data)
model_cnn.eval()
with torch.no_grad():
    
    x, y, pose = next(dtrain)
    pose = pose.to(device)
    y_pred = model_cnn(x.to(device))
    y = y.to(device)
    
    pose0 = PT.from_lie(pose)
    trajectory = TT.from_lie_relative(y_pred, pose0[0])
    trajectory_fact = TT.from_absolute(pose0)
    
    dataset_train = tqdm(dtrain, desc=f'Поехали', position=0)
    for item in dataset_train:
        x, y, _ = item
        
        y_pred = model_cnn(x.to(device))
        trajectory.extend_lie_relative(y_pred)
        trajectory_fact.extend_lie_relative(y)
        
trajectory_fact = trajectory_fact.positions().cpu()
trajectory = trajectory[:-1].positions().cpu()

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