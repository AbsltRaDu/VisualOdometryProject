import  random
import json
import os

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import Subset

from src.dataloaders.dataloader_for_CNN import mavDataLoader, SequenceBatchSampler
from src.dataloaders.dataloader_for_RCNN import dataloaderRCNN
from src.models.DeepVO import PairwiseVOModel, DeepVORNN
from src.function_of_loss.mse_pose import PoseLoss
from src.piplines.pipline_for_RCNN import training
from src.geometry.get_tensors_of_x_y import get_tensors_of_x_y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])
# []
dataset = mavDataLoader('datasets/euroc_mav', transform, device='cpu', batchsize=16, lst_of_datasets=['mav0_easy1', 'mav0_vic1', 'mav0_vic2', 'mav0_easy2', 'mav0_dif1', 'mav0_dif2']) # Сразу формируем все массивы на GPU

groups = dataset.batch_groups.copy()
dataset_groups = dataset.dataset_group

sampler = SequenceBatchSampler(groups)

train_data = data.DataLoader(dataset, batch_sampler=sampler, num_workers=6, pin_memory=True)
print('Длина датасета:', len(dataset), sep=' ')
example_of_obj = next(iter(dataset))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
print('Размерность T_m:', example_of_obj[2].shape, sep=' ')

state_dict = torch.load("process_of_fitting/fitting_models/CNN_for_RCNN_1_0.tar", map_location="cpu")

model = PairwiseVOModel()
model.load_state_dict(state_dict)
model_cnn = model.encoder.to(device)

seq_len = 64
hidden_size = 1000
model_rcnn = DeepVORNN(hidden_size=hidden_size)
model_rcnn = model_rcnn.to(device)

tensor_of_x, tensor_of_y, tensor_of_T = get_tensors_of_x_y(train_data, model_cnn, device) # Получаем вектора представлений x CNN и единый тензор y
dataset_r = dataloaderRCNN(tensor_of_x, tensor_of_y, seq_len, dataset_groups)
train_data = data.DataLoader(dataset_r, batch_size=8, num_workers=0, pin_memory=True)
test_data = data.DataLoader(dataset_r, batch_size=8, num_workers=0, pin_memory=True)

print('Длина датасета c последовательностями:', len(dataset_r), sep=' ')
example_of_obj = next(iter(dataset_r))
print('Размерность X:', example_of_obj[0].shape, sep=' ')
print('Размерность y:', example_of_obj[1].shape, sep=' ')
# print('Размерность T_m:', example_of_obj[2].shape, sep=' ')

epochs = 10
loss_func = PoseLoss()
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model_rcnn.parameters()), lr=1e-4)
count_of_params = sum(p.numel() for p in model_rcnn.parameters() if p.requires_grad)
print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

# Процесс обучения
dct_of_results = training(train_data, test_data, model_rcnn, loss_func, optimizer, epochs, device=device, name_of_model=os.path.join('process_of_fitting/fitting_models', 'DeepVO.tar'), squueze=False)

with open(os.path.join('process_of_fitting/result_of_fitting', 'DeepVO.json'), 'w', encoding='utf-8') as w:
   json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
    