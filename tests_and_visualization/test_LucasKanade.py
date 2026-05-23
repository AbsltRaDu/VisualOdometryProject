import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.dataloaders.datasets_KITTI import kittiDataset_3D
from src.models.modelsClassic.LucaseKanade import LKOpticalFlowVO
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray
from src.piplines.SimulationPipline import SimulationClassic


transform = T.Compose([
    # T.Resize((192, 320)),
    TorchImageToCvGray()
])

normalize = None

lst_of_dataset = [i for i in os.listdir('datasets/sequences') if i != 'poses']
lst_of_dataset_train = lst_of_dataset[1:]
lst_of_dataset_test = lst_of_dataset[0]

dataset = kittiDataset_3D(
    'datasets/sequences',
    'datasets/sequences/poses',
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=False
    )

# lst_of_dataset = os.listdir('datasets/simulation')
# lst_of_dataset_test = ['mav_altitude_wave_500m']

# dataset = mavDatasetCNN_3D(
#     'datasets/simulation', 
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=True,
#     )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')


model = LKOpticalFlowVO(width=752, height=480, new_width=752, new_height=480, fov_deg=90, baseline=0.2)

dtrain = iter(dtrain)

print(next(dtrain)[0].shape)

simulation = SimulationClassic(model=model, device='cpu', dtrain=dtrain, norm=normalize)
simulation()
simulation.get_pictures()