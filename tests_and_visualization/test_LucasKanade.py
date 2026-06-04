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

# lst_of_dataset = [i for i in os.listdir('datasets/sequences') if i != 'poses']
# lst_of_dataset_train = lst_of_dataset[1:]
# lst_of_dataset_test = lst_of_dataset[0]

# dataset = kittiDataset_3D(
#     'datasets/sequences',
#     'datasets/sequences/poses',
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=True
#     )

# lst_of_dataset = os.listdir('datasets/simulation')
# lst_of_dataset_test = ['mav_square']

# dataset = mavDatasetCNN_3D(
#     'datasets/simulation', 
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=True,
#     )

lst_of_dataset = os.listdir('datasets/pok')
lst_of_dataset_test = ['mav_stationary_hover_60s']
# lst_of_dataset_test = ['mav_mixed_random_short_500m']

dataset = mavDatasetCNN_3D(
    'datasets/pok', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=True
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')


model = LKOpticalFlowVO(width=1241, height=376, new_width=1241, new_height=376, fov_deg=90, baseline=0.6, return_debug=True, max_depth=500, min_depth=0.1)

dtrain = iter(dtrain)

print(next(dtrain)[0].shape)

simulation = SimulationClassic(model=model, device='cpu', dtrain=dtrain, norm=normalize, debug=True, path_file_of_result='tests_and_visualization/results_of_models/LK2.json', win_size=100)
simulation()
simulation.get_pictures()