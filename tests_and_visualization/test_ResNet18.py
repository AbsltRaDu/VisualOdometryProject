import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.dataloaders.datasets_KITTI import kittiDataset_3D
from src.models.modelsNN.pairwaisNN.CNN_ResNet18_VO import CNN_ResNet18_VO
from src.piplines.SimulationPipline import SimulationCNN

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

# lst_of_dataset = os.listdir('datasets/simulation')
# lst_of_dataset_test = ['mav_square']

# dataset = mavDatasetCNN_3D(
#     'datasets/simulation', 
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=True
#     )

lst_of_dataset = [i for i in os.listdir('datasets/sequences') if i != 'poses']
lst_of_dataset_test = lst_of_dataset[0]

dataset = kittiDataset_3D(
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

model = CNN_ResNet18_VO()
state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet18_KITTI.tar', map_location=device)
model.load_state_dict(state_dict_cnn)
model = model.to(device)

dtrain = iter(dtrain)

simulation = SimulationCNN(model=model, device=device, dtrain=dtrain, norm=normalize)
simulation()
simulation.get_pictures()