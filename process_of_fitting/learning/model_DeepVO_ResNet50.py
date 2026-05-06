import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
from src.dataloaders.Samplers import BatchSeqSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.models.CNN_ResNet18_VO import CNN_ResNet18_VO
from src.models.DeepVO_ResNet50 import DeepVO_ResNet50
from src.function_of_loss.mse_pose import PoseLoseSeq, PoseLossTrajectorySeq
from src.piplines.pipline_learning_NN import training_RCNN_JointTraning

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

lst_of_datasets = ['mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']
lst_of_datasets_for_train = ['mav0_easy2']
lst_of_datasets_for_test = ['mav0_easy1']

dataset_train = mavDatasetCNN_3D('datasets/euroc_mav', transform, normalize=None, device='cpu', lst_of_datasets=lst_of_datasets) # Сразу формируем все массивы на GPU
dataset_test = mavDatasetCNN_3D('datasets/euroc_mav', transform, normalize=None, device='cpu', lst_of_datasets=lst_of_datasets_for_test)

dataset_train = SequenceDataset(dataset_train, seq=10)
dataset_test = SequenceDataset(dataset_test, seq=10)

train_sampler = BatchSeqSampler(dataset_train, batch_size=8)
test_sampler = BatchSeqSampler(dataset_test, batch_size=8)

train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True)
test_data = data.DataLoader(dataset_test, batch_sampler=test_sampler, num_workers=6, pin_memory=True)

print('Длина тренировочного выборки:', len(dataset_train), sep=' ')
print('Длина тестовой выборки:', len(dataset_test), sep=' ')

example_of_obj = next(iter(train_data))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
print('Размерность T_m:', example_of_obj[2].shape, sep=' ')

encoder = CNN_ResNet50_VO()
model = DeepVO_ResNet50(encoder=encoder, feat_dim=2048, hidden_size=1024)

if os.path.isfile('process_of_fitting/fitting_models/DeepVO_ResNet50_VO_GL.tar'):
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/DeepVO_ResNet50_VO_GL.tar')
    model.load_state_dict(state_dict_cnn)
    print('Были загружены веса модели с контрольной точки')

else:
    state_dict_cnn = torch.load('process_of_fitting/fitting_models/CNNResNet50_VO.tar')
    encoder.load_state_dict(state_dict_cnn)
    model = DeepVO_ResNet50(encoder=encoder)
    print('Были загружены веса экодера')

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
loss_func = PoseLoseSeq(k=1) 
loss_func_trajectory = PoseLossTrajectorySeq()
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-8)
count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_RCNN_JointTraning(train_data, test_data, model, loss_func_pose=loss_func, loss_func_trajectory=loss_func_trajectory, optimizer=optimizer, \
    epochs=epochs, device=device, normalize=None, \
    name_of_model=os.path.join('process_of_fitting/fitting_models', 'DeepVO_ResNet50_VO_GL.tar'), path_to_save_process_of_fitting=os.path.join('process_of_fitting/result_of_fitting', 'DeepVO_ResNet50_VO_GL.json'), squueze=False, \
        weigth_local=1, weigth_trajectory=0.4, accum_steps=100)

