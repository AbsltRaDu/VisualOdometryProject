import torch
from torch.utils import data
from torchvision import transforms as T
import os
from torchvision.models.optical_flow import raft_small

from dataloaders.datasets import mavDatasetCNN_RAFT
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie


from models_from_github.liteflownet.run import Network



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Обучение на:', device, sep=' ')

transform = T.Compose([
    T.Resize((192, 320)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

normalize = None

lst_of_dataset = os.listdir('datasets/simulation')
lst_of_dataset_train = lst_of_dataset[:-1]
lst_of_dataset_test = lst_of_dataset[-1]

model = Network()
model = model.to(device)

print(model)


