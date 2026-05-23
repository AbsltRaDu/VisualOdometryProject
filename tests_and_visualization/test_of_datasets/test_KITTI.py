import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_KITTI import kittiDataset_3D_euler
from src.piplines.SimulationPipline import SimulationCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')
    
transform = T.Compose([
    T.Resize((192, 640)),
    T.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    )
])

normalize = None

lst_of_dataset = os.listdir('datasets/sequences')
lst_of_dataset_test = ['00']

dataset = kittiDataset_3D_euler(
    'datasets/sequences',
    'datasets/sequences/poses',
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=True
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

dtrain = iter(dtrain)

simulation = SimulationCNN(model=None, device=device, dtrain=dtrain, norm=normalize, visualization=True)
simulation()
simulation.get_pictures()