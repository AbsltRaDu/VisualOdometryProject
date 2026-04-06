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
from src.piplines.pipline_for_CNN import training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')


transform = T.Compose([
    T.Resize((320, 192))
])
dataset = mavDataLoader('datasets/euroc_mav', transform, device='cpu', batchsize=4, hidden_size=8, lst_of_datasets=['mav0_easy1']) # Сразу формируем все массивы на GPU

train_size = int(0.005 * len(dataset))

groups = dataset.batch_groups.copy()

train_size = int(0.8 * len(groups))
train_groups = groups[:train_size]
test_groups = groups[train_size:]


train_sampler = SequenceBatchSampler(train_groups)
test_sampler = SequenceBatchSampler(test_groups)

train_data = data.DataLoader(dataset, batch_sampler=train_sampler, num_workers=6, pin_memory=True)
test_data = data.DataLoader(dataset, batch_sampler=test_sampler, num_workers=6, pin_memory=True)
print('Длина датасета:', len(dataset), sep=' ')

model = DeepVO()
model = model.to(device)

epochs = 10
loss_func = PoseLoss()
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
count_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Кол-во обучаемых параметров модели:', count_of_params, sep=' ')

# Процесс обучения
dct_of_results = training(train_data, test_data, model, loss_func, optimizer, epochs, device=device, name_of_model=os.path.join('process_of_fitting/fitting_models', 'DeepVO.tar'), squueze=False)

with open(os.path.join('process_of_fitting/result_of_fitting', 'DeepVO.json'), 'w', encoding='utf-8') as w:
   json.dump(dct_of_results, w, ensure_ascii=False, indent=4)

    