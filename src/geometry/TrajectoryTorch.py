import torch
from dataclasses import dataclass
from typing import Optional

from src.geometry.PoseTorch import PoseTorch

@dataclass
class TrajectoryTorch:
    '''
    Класс для работы с траекторией, последовательностью поз SE(3)
    '''
    
    poses: PoseTorch # размерности (..., N, 4, 4), где N - длина последовательности
    
    def __post_init__(self):
        '''
        Проверка корректности формы входного тензора
        '''
        
        if not isinstance(self.poses, PoseTorch):
            raise TypeError('poses должен быть объектом PoseTorch')
        
    @classmethod
    def from_absolute(cls, poses: PoseTorch) -> 'TrajectoryTorch':
        '''
        Создание траектории из уже готовой последовательности абсолютных поз в тензоре торча
        '''
        
        return cls(poses)
    
    @classmethod
    def from_matrices(cls, poses: torch.Tensor) -> 'TrajectoryTorch':
        '''
        Создание траектории из тензора абсолютных матриц поз
        '''
        
        return cls(PoseTorch.from_matrix(poses))
        
    @classmethod
    def from_relative(cls, deltas: PoseTorch, pose0: Optional[PoseTorch] = None) -> 'TrajectoryTorch':
        '''
        Создание абсолютной траектории из последовательности относительного движения
        '''
        
        if not isinstance(deltas, PoseTorch):
            raise TypeError('deltas должен быть объектом PoseTorch')
        
        # TODO добавить проверку, что shape PoseTorch > 1. Так как класс TrajectoryTorch будет работать только с последовательностями, налогично torch.Tensor
        
        batch_shape = deltas.t.shape[:-2]
        num_deltas = deltas.t.shape[-2]
        
            
        device = deltas.device
        dtype = deltas.dtype
        
        if pose0 is None:
            # Создаем единичную позу для каждой батчевой размерности
            R0 = deltas.R.identity(batch_shape=batch_shape, device=device, dtype=dtype)
            t0 = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
            pose0 = PoseTorch.from_rt(R0, t0)
        
        poses_list = [pose0]
        current = pose0
        
        # Блок накопления траектории

        for k in range(num_deltas):
            current = current * deltas[k] # Берем k с уровня num_deltas позу. В случае, когда у нас и батчи и последовательность, мы берем сразу несколько поз из последовательности с разных батчей и умнажаем на общую стартовую
            poses_list.append(current)
            
        poses = PoseTorch.stack(poses_list, dim=-2)
        
        return cls(poses)