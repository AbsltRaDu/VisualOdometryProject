import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetGPS
from src.models.modelsINS.PureINS import PureINSModel, IMUPreprocessor, INSPropagator
from src.models.Filters.UKF import SigmaPointInertialFusionFilter, UKFNoiseConfig
from src.models.modelsClassic.LucaseKanade import LKOpticalFlowVO
from src.models.modelsClassic.blocks.transforms_for_classic import TorchImageToCvGray
from src.piplines.SimulationPipline import SimulationUFK

from src.geometry.PoseTorch import PoseTorch as PT
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')
device = 'cpu'
    
    
transform = T.Compose([
    # T.Resize((192, 320)),
    TorchImageToCvGray()
])

normalize = None

lst_of_dataset = os.listdir('datasets/euroc_mav')
lst_of_dataset_test = ['mav0_vic1']

dataset = mavDatasetGPS(
    'datasets/euroc_mav', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    num_of_imu=None,
    get_imu_t=True,
    stereo=True,
    end=1000
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

K0 = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
], dtype=np.float64)

dist0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float64)

x, imu, imu_t, y, pose, gps = next(iter(dtrain))

pose = PT.from_lie(pose)


ukf = SigmaPointInertialFusionFilter(
    init_pose=pose,
    init_velocity=torch.zeros(3, dtype=torch.float64),
    gravity_xyz=(0.0, 0.0, 9.81),
    device="cpu",
    dtype=torch.float64,
    # noise=config
)

modelVO = LKOpticalFlowVO(width=752, height=480, new_width=752, new_height=480, baseline=0.11, 
                        fov_deg=None, K=K0, dist=dist0,
                        return_debug=False, max_depth=500, min_depth=0.1)



dtrain = iter(dtrain)


simulation = SimulationUFK(UKF=ukf, model=modelVO, device='cpu', dtrain=dtrain, norm=normalize, debug=False, 
                           path_file_of_result='tests_and_visualization/results_of_models/UFK1.json', gps_window=0, win_size=1)
simulation()
simulation.get_pictures()

