import os

import torch
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D
from src.dataloaders.datasets_KITTI import kittiDataset_3D_euler
from src.piplines.SimulationPipline import SimulationCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')
    
transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    )
])

normalize = None

# lst_of_dataset = os.listdir('datasets/sequences')
# lst_of_dataset_test = ['00']

# dataset = kittiDataset_3D_euler(
#     'datasets/sequences',
#     'datasets/sequences/poses',
#     transform,
#     normalize=normalize,
#     device='cpu',
#     lst_of_datasets=lst_of_dataset_test,
#     stereo=True
#     )

lst_of_dataset = os.listdir('datasets/simulation_2')
lst_of_dataset_test = ['mav_look_forward_rectangle_500m', 
                       'mav_look_forward_climb_soft_turns_450m', 
                       'mav_look_forward_soft_s_curve_500m', 
                       'mav_look_forward_long_soft_zigzag_altitude_750m',
                       'mav_look_forward_descent_soft_turns_450m']
lst_of_dataset_train = [i for i in lst_of_dataset if i not in lst_of_dataset_test]
lst_of_dataset_test = ['box']



dataset = mavDatasetCNN_3D(
    'datasets/simulation_2', 
    transform,
    normalize=normalize,
    device='cpu',
    lst_of_datasets=lst_of_dataset_test,
    stereo=True
    ) 

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


dtrain = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
print('Длина датасета:', len(dataset), sep=' ')

dtrain = iter(dtrain)

simulation = SimulationCNN(model=None, device=device, dtrain=dtrain, norm=normalize, visualization=True, path_file_of_result='tests_and_visualization/results_of_models/testKITTI.json')
simulation()
simulation.get_pictures()