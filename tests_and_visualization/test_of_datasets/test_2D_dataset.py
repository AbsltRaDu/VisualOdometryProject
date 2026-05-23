import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets2D import mavDatasetCNN_2D
from src.piplines.SimulationPipline import SimulationCNN2D


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')
    
transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_test = ['mav_square']

dataset = mavDatasetCNN_2D(
    'datasets/simulation', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=True
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

dtrain = iter(dtrain)

simulation = SimulationCNN2D(model=None, device=device, dtrain=dtrain, norm=normalize, visualization=True)
simulation()
simulation.get_pictures()