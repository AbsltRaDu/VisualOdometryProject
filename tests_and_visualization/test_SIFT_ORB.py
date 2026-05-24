import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.models.modelsClassic.FMM import FeatureStereoVOConfig, StereoCameraConfig, CV2FeatureStereoVO
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray
from src.piplines.SimulationPipline import SimulationClassic


transform = T.Compose([
    # T.Resize((192, 320)),
    TorchImageToCvGray()
])

normalize = None


lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_test = ['mav_square']

dataset = mavDatasetCNN_3D(
    'datasets/simulation', 
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
        width=752,
        height=480,
        new_width=752,
        new_height=480,
        fov_deg=90.0,
        baseline=0.2,
    ),
    feature_type="orb",
    nfeatures=8000,
    ratio_test=0.95,
    return_debug=False,
)

model = CV2FeatureStereoVO(config)
dtrain = iter(dtrain)

print(next(dtrain)[0].shape)

simulation = SimulationClassic(model=model, device='cpu', dtrain=dtrain, norm=normalize)
simulation()
simulation.get_pictures()