import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.models.modelsClassic.FMM import FeatureStereoVOConfig, StereoCameraConfig, CV2FeatureStereoVO
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray
from src.piplines.SimulationPipline import SimulationClassic


transform = T.Compose([
    # T.Resize((192, 640)),
    TorchImageToCvGray()
])

normalize = None


lst_of_dataset = os.listdir('datasets/simulation_2')
lst_of_dataset_test = ['mav_look_forward_climb_soft_turns_450m']

dataset = mavDatasetCNN_3D(
    'datasets/simulation_2', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=True,
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')


config = FeatureStereoVOConfig(
    camera=StereoCameraConfig(
        width=1241,
        height=376,
        new_width=1241,
        new_height=376,
        fov_deg=90.0,
        baseline=0.6,
    ),
    feature_type="orb",
    nfeatures=8000,
    ratio_test=0.75,
    return_debug=False,
    max_depth=500, min_depth=0.1
)

model = CV2FeatureStereoVO(config)
dtrain = iter(dtrain)

print(next(dtrain)[0].shape)

simulation = SimulationClassic(model=model, device='cpu', dtrain=dtrain, norm=normalize, visualization=False, path_file_of_result='tests_and_visualization/results_of_models/ORB1.json', win_size=100)
simulation()
simulation.get_pictures()