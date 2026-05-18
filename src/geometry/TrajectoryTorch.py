import torch
from dataclasses import dataclass
from typing import Optional

from src.geometry.PoseTorch import PoseTorch
from src.geometry.RotationTorch import RotationTorch

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
    def from_matrices_absolute(cls, poses: torch.Tensor) -> 'TrajectoryTorch':
        '''
        Создание траектории из тензора абсолютных матриц поз
        '''
        
        return cls(PoseTorch.from_matrix(poses))
    
    @classmethod
    def from_lie_absolute(cls, poses: torch.Tensor) -> 'TrajectoryTorch':
        '''
        Создание траектории из тензора абсолютных поз представленных алгеброй Ли
        '''
        
        return cls(PoseTorch.from_lie(poses))
        
    @classmethod
    def from_relative(cls, deltas: PoseTorch, pose0: Optional[PoseTorch] = None) -> 'TrajectoryTorch':
        '''
        Создание абсолютной траектории из последовательности относительного движения
        '''
        # TODO продумать, как обрабатывать PoseTorch.ndim = 1
        
        if not isinstance(deltas, PoseTorch):
            raise TypeError('deltas должен быть объектом PoseTorch')
        
        # TODO добавить проверку, что shape PoseTorch > 1. Так как класс TrajectoryTorch будет работать только с последовательностями, налогично torch.Tensor
        
        if deltas.t.ndim >= 2:
            batch_shape = deltas.t.shape[:-2]
            num_deltas = deltas.t.shape[-2]
        
        else:
            batch_shape = ()
            num_deltas = 0

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

        if num_deltas > 0:
        
            for k in range(num_deltas):
                current = current * deltas[k] # Берем k с уровня num_deltas позу. В случае, когда у нас и батчи и последовательность, мы берем сразу несколько поз из последовательности с разных батчей и умнажаем на общую стартовую
                poses_list.append(current)
                
            poses = PoseTorch.stack(poses_list, dim=-2)
            
        else:
            current = current * deltas
            poses_list.append(current)
            
            poses = PoseTorch.stack(poses_list, dim=0)
        
        return cls(poses)

    @classmethod
    def from_matrices_relative(cls, poses: torch.Tensor, pose0: Optional[PoseTorch] = None) -> 'TrajectoryTorch':
        '''
        Создание траектории из последовательности относительных движений, представленных матрицами движения
        '''
        
        deltas = PoseTorch.from_matrix(poses)
        return cls.from_relative(deltas, pose0)
        
    @classmethod
    def from_lie_relative(cls, xi_seq: torch.Tensor, pose0: Optional[PoseTorch] = None) -> 'TrajectoryTorch':
        '''
        Создание траектории из последовательности относительных движений, представленных алгеброй Ли
        '''
        
        deltas = PoseTorch.from_lie(xi_seq)
        return cls.from_relative(deltas, pose0)
    
    
    
    def extend_relative(self, deltas: PoseTorch, keep_history: bool = True) -> 'TrajectoryTorch':
        '''
        Продолжает накполенную траектори, добавляя относительные преобразования
        '''
        
        if not isinstance(deltas, PoseTorch):
            raise TypeError('deltas должен быть объектом PoseTorch')
        
        last_pose = self.last_pose()
        
        if keep_history:
            trajectory = self.from_relative(deltas, last_pose)
            trajectory = TrajectoryTorch.cat([TrajectoryTorch.from_absolute(self.poses[:-1]), trajectory], dim=-2)
        else:
            trajectory = self.from_relative(deltas, last_pose)
        
        return trajectory
    
    def extend_martices_relative(self, poses: torch.Tensor, keep_history: bool = True) -> 'TrajectoryTorch':
        '''
        Продолжает накполенную траектори, добавляя относительные преобразования в матричном представлении
        '''
        
        deltas = PoseTorch.from_matrix(poses)
        
        return self.extend_relative(deltas, keep_history)
    
    def extend_lie_relative(self, xi_seq: torch.Tensor, keep_history: bool = True) -> 'TrajectoryTorch':
        '''
        Продолжает накполенную траектори, добавляя относительные преобразования в матричном представлении
        '''
        
        deltas = PoseTorch.from_lie(xi_seq)
        
        return self.extend_relative(deltas, keep_history)
    
    def as_pose(self) -> PoseTorch:
        '''
        Возвращает траекторию, как объекты PoseTorch
        '''
        
        return self.poses.clone()
    
    def as_matrices(self) -> torch.Tensor:
        '''
        Возвращает траекторию, как тензор мамтриц поз
        '''
        
        return self.poses.as_matrix()
        
    
    def as_lie(self) -> torch.Tensor:
        '''
        Возвращает траекторию, в представлении алгебры Ли
        '''
        
        return self.poses.as_lie()
        
        
    def positions(self) -> torch.Tensor:
        '''
        Возвращает только координаты центров поз
        '''
        
        return self.poses.translation()
    
    def rotations(self) -> RotationTorch:
        '''
        Вовзращает объекты вращений RotationTorch
        '''
        
        return self.poses.rotation()
    
    def relative_deltas(self) -> PoseTorch:
        '''
        Вычисляет последовательность относительных движений между соседними позами
        '''
        
        deltas = [pose0.inv() * pose1 for pose0, pose1 in  zip(self.poses[:-1], self.poses[1:])]
        
        return PoseTorch.stack(deltas, dim=-2)
    
    def relative_motion(self, start_index: int = 0, end_index: int = -1) -> PoseTorch:
        '''
        Возвращает матрицу сдвига от начальной позы к конечной
        
        метод сугубо для метрики KITTI
        '''
    
        deltas = self.poses[start_index].inv() * self.poses[end_index]
        
        return deltas
    
    def path_length(self) -> torch.Tensor:
        '''
        Вычисляет суммарную длину траектории по последовательности поз
        '''
        
        pos = self.positions()
        diffs = pos[..., 1:, :] - pos[..., :-1, :]
        seq_lengths = torch.linalg.norm(diffs, dim=-1)
        
        return seq_lengths.sum(dim=-1)
    
    def first_pose(self) -> PoseTorch:
        '''
        Возвращает первую позу траектории, как объект PoseTorch
        '''
    
        return self.poses[0]
    
    def last_pose(self) -> PoseTorch:
        '''
        Возвращает последнюю позу траектории, как объект PoseTorch
        '''
        
        return self.poses[-1]
    
    def clone(self ) -> 'TrajectoryTorch':
        '''
        Создает глубокую копию объекта
        '''

        return TrajectoryTorch(self.poses.clone())
    
    def detach(self) -> 'TrajectoryTorch':
        '''
        Отсоединяет траекторию от графа вычислений
        '''
        
        return TrajectoryTorch(self.poses.detach())
    
    def to(self, *args, **kwargs) -> 'TrajectoryTorch':
        '''
        Переводит траекторию на другой девай dtype
        '''
        
        return TrajectoryTorch(self.poses.to(*args, **kwargs))
        
    @classmethod
    def stack(cls, poses: list['TrajectoryTorch'], dim: int = 0) -> 'TrajectoryTorch':
        '''
        Аналог torch.stack для объектов TrajectoryTorch
        '''
        
        if len(poses) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка TrajectoryTorch')
        
        if not all(isinstance(p, TrajectoryTorch) for p in poses):
           raise TypeError('Все элементы списка должны быть объектами TrajectoryTorch') 
        
        return cls(PoseTorch.stack([traj.poses for traj in poses], dim=dim))
    
    @classmethod
    def cat(cls, poses: list['TrajectoryTorch'], dim: int = 0) -> 'TrajectoryTorch':
        '''
        Аналог torch.stack для объектов TrajectoryTorch
        '''
        
        if len(poses) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка TrajectoryTorch')
        
        if not all(isinstance(p, TrajectoryTorch) for p in poses):
           raise TypeError('Все элементы списка должны быть объектами TrajectoryTorch') 
        
        return cls(PoseTorch.cat([traj.poses for traj in poses], dim=dim))
    
    @property
    def device(self):
        return self.poses.device

    @property
    def dtype(self):
        return self.poses.dtype
    
    @property
    def shape(self):
        '''
        Возвращает батчевую форму траектории, без размерностей (N, 3)
        '''

        return self.poses.t.shape[:-2]
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, item):
        
        if self.poses.t.ndim >= 2:
        
            pose = self.poses[item]
            
            return TrajectoryTorch.from_absolute(pose)
        
        else:
            raise ValueError('Данный объект Trajectory не итерируемый')