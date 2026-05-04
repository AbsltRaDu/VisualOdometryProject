import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple
from collections import deque

import torch
from torch.utils import data

class BatchSampler(data.Sampler):
    '''
    Простой сэмплер для обычного обучения RNN
    '''
    
    def __init__(self, dataset: data.Dataset, batch_size: int, shuffle: bool = False, drop_last: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        
        indeces_lst = self.dataset.dataset_group
        
        for indeces in indeces_lst:
            batch = []
            
            for idx in range(*indeces):
                
                batch.append(idx)
            
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                
            if len(batch) > 0 and not self.drop_last:
                yield batch
                
    def __len__(self):
        
        if self.drop_last:
            return sum([(idxs[1] - idxs[0]) // self.batch_size for idxs in self.dataset.dataset_group])
        
        return sum([((idxs[1] - idxs[0]) + self.batch_size - 1) // self.batch_size for idxs in self.dataset.dataset_group])
    
class BatchSeqSampler(BatchSampler):
    '''
    Семплер для обертки над датасетом с учетом последовательностей
    '''
    
    
    
    def __init__(self, dataset: data.Dataset, batch_size: int, shuffle: bool = False, drop_last: bool = True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
    def __iter__(self):
        
        indeces_lst = self.dataset.seq_groups
        batch = []
        
        for indeces in indeces_lst:
            for idx in range(*indeces):
                batch.append(idx)
            
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            
        if len(batch) > 0 and not self.drop_last:
                yield batch
                
    def __len__(self):
        
        if self.drop_last:
            return sum([(idxs[1] - idxs[0]) // self.batch_size for idxs in self.dataset.seq_groups])
        
        return sum([((idxs[1] - idxs[0]) + self.batch_size - 1) // self.batch_size for idxs in self.dataset.seq_groups])

                
    

                
        
        
        