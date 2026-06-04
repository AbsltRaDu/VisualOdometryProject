import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetVIO
from src.models.modelsINS.PureINS import PureINSModel, IMUPreprocessor, INSPropagator
from src.piplines.SimulationPipline import SimulationIMU

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

lst_of_dataset = os.listdir('datasets/euroc_mav')
lst_of_dataset_test = ['mav0_dif1']

dataset = mavDatasetVIO(
    'datasets/euroc_mav', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    num_of_imu=None,
    get_imu_t=True,
    stereo=True,
    end=200
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

propagator = INSPropagator(gravity=torch.tensor([0, 0, 9.81]), use_updated_rotation_for_accel=False)
preprocessor = IMUPreprocessor()
model = PureINSModel(preprocessor, propagator)

dtrain = iter(dtrain)

simulation = SimulationIMU(model=model, device='cpu', dtrain=dtrain, norm=normalize)
simulation()
simulation.get_pictures()