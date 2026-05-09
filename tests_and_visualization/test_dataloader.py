import torch
from torch.utils import data
from torchvision import transforms as T
import os
from torchvision.models.optical_flow import raft_small

from src.dataloaders.datasets_for_CNN_RAFT import mavDatasetCNN_RAFT
from src.dataloaders.Samplers import ProgressiveWindowBatchSampler
from src.normalize.PoseNormolizerLie import PoseNormalizerLie

from src.dataloaders.RAFTFeaturesDataset import RAFTFeaturesDataset
from src.models.exportEncoderFeatures import export_encoder_features_to_csv
from src.models.CNN_RAFT import RAFTPoseCNN

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

model = RAFTPoseCNN()
model = model.to(device)

dataset_train = mavDatasetCNN_RAFT('datasets/simulation', transform, normalize=normalize, device='cpu', lst_of_datasets=lst_of_dataset_test) 
train_sampler = ProgressiveWindowBatchSampler(dataset_train, batch_size=32, window_size=1, shuffle=False)
train_data = data.DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=6, pin_memory=True)

csv_path = 'process_of_fitting/transfer_learning/dataset_features.csv'
export_encoder_features_to_csv(encoder=model.flow_model, dataset=train_data, ex=dataset_train[0], csv_path=csv_path, device=device)