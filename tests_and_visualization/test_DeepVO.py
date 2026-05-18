import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.models.modelsNN.DeepVO import DeepVO, VisualEncoder
from src.piplines.SimulationPipline import SimulationRNN

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

dataset = mavDatasetCNN_3D(
    'datasets/simulation', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=False
    )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

encoder = VisualEncoder()
model = DeepVO(VisualEncoder=encoder)
state_dict_cnn = torch.load('process_of_fitting/fitting_models/DeepVO.tar', map_location=device)
model.load_state_dict(state_dict_cnn)
model = model.to(device)

dtrain = iter(dtrain)

simulation = SimulationRNN(model=model, device=device, dtrain=dtrain, norm=normalize)
simulation()
simulation.get_pictures()