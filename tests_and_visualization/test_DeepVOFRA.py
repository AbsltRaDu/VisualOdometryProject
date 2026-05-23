import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.dataloaders.datasets_KITTI import kittiDataset_3D
from src.models.modelsNN.DeepVOFRA import DeepVO
from src.piplines.SimulationPipline import SimulationDeepVO
from src.function_of_loss.mse_pose import PoseLoss

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

lst_of_dataset = [i for i in os.listdir('datasets/sequences') if i != 'poses']
lst_of_dataset_train = lst_of_dataset[1:]
lst_of_dataset_test = ['03']

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
# lst_of_dataset_test = ['mav_square']

# dataset = mavDatasetCNN_3D(
#     'datasets/simulation', 
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=False
#     )

dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

model = DeepVO()
model = model.to(device)

# path = "process_of_fitting/fitting_models/checkpoint_e190.pth"
# checkpoint = torch.load(path, map_location="cpu", weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])

state_dict_cnn = torch.load('process_of_fitting/fitting_models/DeepVO_AirSim.tar', map_location=device)
model.load_state_dict(state_dict_cnn)
model = model.to(device)

dtrain = iter(dtrain)
loss_func = PoseLoss()

simulation = SimulationDeepVO(model=model, device=device, dtrain=dtrain, norm=normalize, loss=loss_func)
simulation()
simulation.get_pictures()