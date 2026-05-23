import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets2D import mavDatasetCNN_2D
from src.piplines.SimulationPipline import SimulationClassic2D
from src.models.modelsClassic.LucaseKanade import LKOpticalFlowVO2D2
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')
    
transform = T.Compose([
    TorchImageToCvGray()
])

normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_test = ['al']

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

model = LKOpticalFlowVO2D2(width=752, height=480, new_width=752, new_height=480, fov_deg=90, baseline=0.2)

simulation = SimulationClassic2D(model=model, device='cpu', dtrain=dtrain, norm=normalize, visualization=True)
simulation()
simulation.get_pictures()