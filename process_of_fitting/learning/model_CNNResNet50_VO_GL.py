import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.dataloader_for_CNN import mavDatasetCNN_3D, SequenceBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.function_of_loss.mse_pose import PoseLoss, PoseLossTrajectory
from src.piplines.pipline_learning_NN import training_CNN_JointTraning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Обучение на:', torch.cuda.get_device_name(torch.cuda.current_device()), sep=' ')
else:
    print('Обучение на', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

# TODO нормализация ломает обучение по MSE_Pose. Нужно объяснение, почему
# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')

lst_of_datasets = ['mav0_easy1', 'mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']
lst_of_datasets_for_tests = ['mav0_easy1']

dataset = mavDatasetCNN_3D('datasets/euroc_mav', transform, normalize=None, device='cpu', batchsize=32, lst_of_datasets=lst_of_datasets) # Сразу формируем все массивы на GPU

groups = dataset.batch_groups.copy()

# Блок урезания для теста пайплайна
# len_groups = int(0.01 * len(groups))
# groups = groups[:len_groups]

train_size = int(0.8 * len(groups))
train_groups = groups[:train_size]
test_groups = groups[train_size:]

train_sampler = SequenceBatchSampler(train_groups)
test_sampler = SequenceBatchSampler(test_groups)

train_data = data.DataLoader(dataset, batch_sampler=train_sampler, num_workers=6, pin_memory=True)
test_data = data.DataLoader(dataset, batch_sampler=test_sampler, num_workers=6, pin_memory=True)

print('Длина датасета:', len(dataset), sep=' ')
print('Длина тренировочного выборки:', len(train_data), sep=' ')
print('Длина тестовой выборки:', len(test_data), sep=' ')

example_of_obj = next(iter(dataset))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
print('Размерность T_m:', example_of_obj[2].shape, sep=' ')

model = CNN_ResNet50_VO()

if os.path.isfile('process_of_fitting/fitting_models/CNNResNet50_VO.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet50_VO.tar')
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

model = model.to(device)

# TODO Для более точного описания геометрии в признаках нц дообучить все слои, но это не точно
# # всё заморозили
# for p in model.parameters():
#     p.requires_grad = False

# # обучаем новый первый слой
# for p in model.encoder.conv1.parameters():
#     p.requires_grad = True

# # обучаем самый верхний блок resNet
# for p in model.encoder.layer4.parameters():
#     p.requires_grad = True

# # обучаем полносвязки
# for p in model.fc1.parameters():
#     p.requires_grad = True

# for p in model.fc2.parameters():
#     p.requires_grad = True

epochs = 30
# TODO В представлении алгебры Ли значения векторов w, u имею примерно один диопозон. Коэф. к побуждает фокусироваться на корректировки вращений?
loss_func = PoseLoss(k=1) 
loss_func_trajectory = PoseLossTrajectory(reduction='mean')
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-8)
count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_CNN_JointTraning(train_data, test_data, model, loss_func_pose=loss_func, loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, \
    name_of_model=os.path.join('process_of_fitting/fitting_models', 'CNNResNet50_VO_GL.tar'), path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'CNNResNet50_VO_GL_2_0.json'), squueze=False, \
        weigth_local=1, weigth_trajectory=0)

