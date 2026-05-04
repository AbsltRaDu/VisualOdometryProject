import torch
from torch.utils import data
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
from src.dataloaders.Samplers import BatchSampler, BatchSeqSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

normalize = PoseNormalizerLie()
normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')

dataset = mavDatasetCNN_3D('datasets/euroc_mav', transform, normalize=normalize, device='cpu', lst_of_datasets=['mav0_easy1', 'mav0_easy2']) # Сразу формируем все массивы на GPU
dataset_seq = SequenceDataset(dataset, seq=10)
sampler_seq = BatchSeqSampler(dataset_seq, batch_size=32)

dtrain = data.DataLoader(dataset_seq, batch_sampler=sampler_seq)

print(len(dataset_seq))
print(len(dtrain))


dtrain = iter(dtrain)

x, y, z = next(dtrain)
print(x.shape)

print(dataset_seq.seq_groups)
print(dataset.dataset_group)


