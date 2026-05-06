import torch
from torch.utils import data
from torchvision import transforms as T

from src.dataloaders.datasets_for_CNN import mavDatasetCNN_3D, SequenceDataset
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((320, 192))
])

# normalize = PoseNormalizerLie()
# normalize.load('process_of_fitting/normalize_params/params_of_normalize.json')

dataset = mavDatasetCNN_3D('datasets/simulation', transform, normalize=None, device='cpu', lst_of_datasets=['mav0'], max_size=100) # Сразу формируем все массивы на GPU
dataset_seq = SequenceDataset(dataset, seq=10)
sampler_seq = ProgressiveWindowBatchSampler(dataset_seq, batch_size=2, window_size=10, shuffle=True)

# sampler_iterator = iter(sampler_seq)
# b1, b2 = next(sampler_iterator), next(sampler_iterator)
# print(b1, b2, sep='\n')

# print(dataset_seq[b1].shape)

dtrain = data.DataLoader(dataset_seq, batch_sampler=sampler_seq)

print(len(dataset_seq))
print(len(dtrain))


dtrain = iter(dtrain)

x, y, z = next(dtrain)
print(x.shape)

x1, y1, z1 = next(dtrain)


print(torch.cat([x, x1], dim=1).shape)
# print(dataset_seq.seq_groups)
# print(dataset.dataset_group)

