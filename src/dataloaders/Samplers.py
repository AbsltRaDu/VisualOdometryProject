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

class ProgressiveWindowBatchSampler(data.Sampler):
    '''
    Для покадрового смещения с возмодностью оценки траектории
    
    Столкнулся с проблемой, что батч из последовательных смещений не лучший вариант, так как начинает хромать обобщение в сети
    Поэтому этот батч позволяет сохранять последовательность смещений, но уже от батча к батчуу
    в рамках заданного окна window_size. 
    Батчи при этом будут содержать разные отрезки из разных датасетов, что увеличит обощающую способность
    '''
    
    def __init__(self, dataset, batch_size: int, window_size: int, shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.window_starts = [] # Индексы стартов окон
        # Прогнались по датасетам и записали начало и конец с учетом окна
        for start, end in dataset.dataset_group:
            max_start = end - window_size
            for idx in range(start, max_start + 1):
                self.window_starts.append(idx)
                
    def __iter__(self):
        starts = self.window_starts.copy()
        
        if self.shuffle:
            random.shuffle(starts)
            
        for i in range(0, len(starts), self.batch_size):
            batch_starts = starts[i: i+self.batch_size] # По сути сформировали батч начальных значений окошка
            
            if len(batch_starts) < self.batch_size and self.drop_last:
                continue
            
            # А вот теперь погнали итерироваться по окошку
            for step in range(self.window_size):
                batch = [start + step for start in batch_starts]
                yield batch

    def __len__(self):
        num_start_batches = len(self.window_starts) // self.batch_size

        if not self.drop_last and len(self.window_starts) % self.batch_size != 0:
            num_start_batches += 1

        return num_start_batches * self.window_size
    
class BatchSeqSampler(BatchSampler):
    '''
    Семплер для обертки над датасетом с учетом последовательностей и границ датасетов
    '''
    
    
    
    def __init__(self, dataset: data.Dataset, batch_size: int, shuffle: bool = False, drop_last: bool = True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
    def __iter__(self):
        
        indeces_lst = self.dataset.seq_groups
        
        
        for indeces in indeces_lst:
            start, end = indeces
            for i in range(start, end, self.dataset.seq):
                batch = []
                
                for idx in range(i, min(i+self.batch_size, end)):
                    batch.append(idx)
                
                if len(batch) == self.batch_size:
                    yield batch
                
                elif len(batch) > 0 and not self.drop_last:
                    yield batch
                
    def __len__(self):
        total = 0
        
        for indeces in self.dataset.seq_groups:
            start, end = indeces
            for i in range(start, end, self.dataset.seq):
                batch_len = min(i + self.batch_size, end) - i

                if batch_len == self.batch_size:
                    total += 1
                elif batch_len > 0 and not self.drop_last:
                    total += 1

        return total

                
    

                
        
        
        