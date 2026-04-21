import torch
from dataclasses import dataclass

from src.different_functions.RotationTorch import RotationTorch

@dataclass
class PoseTorch:
    '''
    Класс для работы с позой, элементом группы SE(3)
    
    Представляет преобразования вида:
        T = [ R t ]
            [ 0 1 ]
    
    '''
    
    R: RotationTorch # Объект класса ориентации
    t: torch.Tensor # Вектор трансляции размерности (..., 3)
    
    @classmethod
    def from_rt(cls, R: RotationTorch, t: torch.Tensor):
        '''
        Создание позы из вращения и трансляции. где
        
        R - объект вращения
        t - вектор трансляции
        '''
    
        return cls(R, t)
    
    @classmethod
    def from_matrix(cls, T: torch.Tensor):
        '''
        Создание позы из матрицы движения 4 на 4
        '''
        
        R_matrix = T[..., :3, :3]
        t = T[..., :3, 3]
        
        R = RotationTorch.from_matrix(R_matrix)
        
        return cls(R, t)
        
        
    def as_matrix(self) -> torch.Tensor:
        '''
        Возвращает матрицу преобразований SE(3)
        '''
        R_matrix = self.R.as_matrix()
        batch_shape = R_matrix.shape[:-2]
        
        T = torch.zeros(*batch_shape, 4, 4, device=self.t.device, dtype=self.t.dtype)
        T[..., :3, :3] = R_matrix
        T[..., :3, 3] = self.t
        T[..., 3, 3] = 1
        
        return T
    
    def rotation(self) -> RotationTorch:
        '''
        Возвращает объект вращения
        '''
        
        return self.R

    def translation(self) -> torch.Tensor:
        '''
        Возвращает объект трансляции
        '''
        
        return self.t
    
    def inv(self) -> 'PoseTorch':
        '''
        Инверсия позы
        '''
        
        R_inv = self.R.inv()
        t_inv = -R_inv.apply(self.t)
        
        return PoseTorch(R_inv, t_inv)
    
    def compose(self, other: 'PoseTorch') -> 'PoseTorch':
        '''
        Композиция двух поз по формулам:
        
        R = R1 * R2
        t = R1 * t1 + t2
        
        '''
        
        R_new = self.R.compose((other.R)) # выполнение поворота
        
        t_new = self.R.apply(other.t) + self.t # выполнение сдвига
        
        return PoseTorch(R_new, t_new)
        
    def __mul__(self, other: 'PoseTorch') -> 'PoseTorch':
        '''
        Создание оператора умножения
        '''
        
        return self.compose(other)
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        '''
        Выполняет преоразование SE(3) к точкам:
            p' = Rp + t
        '''
        
        return self.R.apply(points) + self.t
    
    def clone(self) -> 'PoseTorch':
        '''
        создание глубокой копии объекта
        '''
        
        return PoseTorch(self.R, self.t.clone())
    
    def detach(self):
        '''
        Отключение взятие производных
        '''
        
        return PoseTorch(self.R, self.t.detach())
    
    def to(self, *args, **kwargs):
        '''
        Переведение на девайс, dtype
        '''
        
        return PoseTorch(self.R, self.t.to(*args, **kwargs))
    
    @property
    def device(self):
        
        return self.t.device
    
    @property
    def dtype(self):
        return self.t.dtype
        
        
        
    