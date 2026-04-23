import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.dataloader_for_CNN import mavDataLoader, SequenceBatchSampler
from src.models.CNN_ResNet50_VO import CNN_ResNet50_VO
from src.models.DeepVO import DeepVO
from src.function_of_loss.mse_pose import PoseLoss
from piplines.pipline_learning_NN import training_CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

lst_of_datasets = ['mav0_easy1', 'mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']
lst_of_datasets_for_tests = ['mav0_easy1']

dataset = mavDataLoader('datasets/euroc_mav', transform, device='cpu', batchsize=16, lst_of_datasets=lst_of_datasets_for_tests) # Сразу формируем все массивы на GPU

groups = dataset.batch_groups.copy()

# Блок урезания для теста пайплайна
len_groups = int(0.01 * len(groups))
groups = groups[:len_groups]

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
model = model.to(device)

# всё заморозили
for p in model.parameters():
    p.requires_grad = False

# обучаем новый первый слой
for p in model.encoder.conv1.parameters():
    p.requires_grad = True

# обучаем самый верхний блок resNet
for p in model.encoder.layer4.parameters():
    p.requires_grad = True

# обучаем полносвязки
for p in model.fc1.parameters():
    p.requires_grad = True

for p in model.fc2.parameters():
    p.requires_grad = True

epochs = 10
loss_func = PoseLoss()
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

dct_of_results = training_CNN(train_data, test_data, model, loss_func, optimizer, epochs, device=device, name_of_model=os.path.join('process_of_fitting/fitting_models', 'CNNResNet50_VO_2_0.tar'), squueze=False)

with open(os.path.join('process_of_fitting/result_of_fitting', 'DeepVO.json')) as w:
   json.dump(dct_of_results, w, ensure_ascii=False, indent=4)