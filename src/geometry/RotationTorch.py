import torch
from dataclasses import dataclass

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from pytorch3d.transforms import quaternion_multiply, quaternion_apply, quaternion_invert

@dataclass
class RotationTorch:
    '''
    Класс для работы с вращениями SO(3) на базе PyTorch. Аналог Rotation из SciPy
    
    Внутреннее представление: кватернион формата (4)
    Конвенция: (w, x, y, z)

    Поддерживает батчевые размерности произвольной формы:
        (4,)           -> одно вращение
        (N, 4)         -> батч вращений
        (B, T, 4)      -> последовательности вращений

    Все вычисления совместимы с CUDA и autograd.    
    '''
    
    _q: torch.Tensor # размерность (4) - кватернион
    
    def __post_init__(self):
        '''
        Метод автоматически вызывается после создания объекта класса dataclass
        '''
        
        if not isinstance(self._q, torch.Tensor):
            raise TypeError('_q должен быть тензором Torch')
        
        if self._q.shape[-1] != 4:
            raise ValueError('Кватернион должен иметь размерность (..., 4)')
        
        self._q = self._normalize_quat(self._q) # нормализация кватерниона
        
    @staticmethod
    def _normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        '''
        Нормализация кватерниона.
        Матрциа вращения задается только единичным кватернионом, 
        поэтому каждый кватернион должен нормализовываться автоматически 
        во избежании проблем с ортогональностью матрицы вращения
        
        q / ||q||
        
        '''
        
        return q / q.norm(dim=-1, keepdim=True).clamp_min(eps) # clamp_min - гарантирует 
    
    @classmethod
    def from_quat(cls, q: torch.Tensor) -> 'RotationTorch':
        '''
        Создание объекта напрямую из кватерниона
        
        Ожидаемый формат: (4)
        '''
        
        return cls(q)
    
    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> 'RotationTorch':
        '''
        Создание вращения из матрицы вращения
        
        Ожидаемый формат: 
        (3, 3)
        '''
        
        q = matrix_to_quaternion(R) # ф-ия преобразования R в кватернион
        return cls(q)
    
    @classmethod
    def from_rotvec(cls, rotvec: torch.Tensor) -> 'RotationTorch':
        '''
        Создание вращения из вектора ось-угол, то есть координат в алгебре Ли so(3)
            rotvec= tetta * u, где
            u - единичный вектор оси вращения
            tetta - угол вращения вокруг этой оси
            
        Ожидаемый формат:
        (3)
        '''
        
        q = axis_angle_to_quaternion(rotvec)
        return cls(q)
    
    @classmethod
    def from_euler(cls, angles: torch.Tensor, seq: str = 'XYZ') -> 'RotationTorch':
        '''
        Создание вращения из углов Эйлера
        
        seq определяет порядок вращений по осям
        
        Ожидаемый формат:
        (3)
        '''
        
        R = euler_angles_to_matrix(angles, convention=seq)
        q = matrix_to_quaternion(R)
        return cls(q)
    
    @classmethod
    def identity(cls, batch_shape=(), device=None, dtype=None) -> 'RotationTorch':
        '''
        Инициализация единичного вращения
        '''
        
        z = torch.zeros(*batch_shape, 4, device=device, dtype=dtype)
        z[..., 0] = 1

        return cls.from_quat(z)
    
    
    def as_quat(self) -> torch.Tensor:
        '''
        Возвращение копии объекта класса-кватерниона в виде кватерниона
        '''
        
        return self._q.clone()
    
    def as_matrix(self) -> torch.Tensor:
        '''
        Возвращение объекта в виде матрицы вращения
        '''
        
        return quaternion_to_matrix(self._q)
    
    def as_rotvec(self) -> torch.Tensor:
        '''
        Возвращение объекта в виде rotvec, то есть координат в алгебре Ли so(3)
        '''
        
        return quaternion_to_axis_angle(self._q)
        
    def as_euler(self, seq: str = 'XYZ') -> torch.Tensor:
        '''
        Возвращение объекта в виде углов Эйлера
        '''
        R = quaternion_to_matrix(self._q)
        
        return matrix_to_euler_angles(R, convention=seq)
    
    def inv(self) -> 'RotationTorch':
        '''
        Инверсия вращения
        
        Для единичного кватеринона обратный равен сопряженному:
            q^-1 = (w, -x, -y, -z)
        '''
    
        q = quaternion_invert(self._q)
        return RotationTorch(q)  
    
    def compose(self, other: 'RotationTorch') -> 'RotationTorch':
        '''
        Композиция двух вращений. Реализуется через умножение кватернионов
        '''
        
        q = quaternion_multiply(self._q, other._q)
        return RotationTorch(q)
    
    def __mul__(self, other: 'RotationTorch') -> 'RotationTorch':
        '''
        Определение поведения оператора * для объектов RotationTorch
        '''
        
        return self.compose(other)
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        '''
        Применение вращения к точкам
        '''
        
        return quaternion_apply(self._q, points)
        
    def to(self, *args, **kwargs) -> 'RotationTorch':
        '''
        Перенос на другой device или dtype
        '''
        
        return RotationTorch(self._q.to(*args, **kwargs))
    
    def clone(self) -> 'RotationTorch':
        '''
        Создание глубокой копии объекта
        '''
        
        return RotationTorch(self._q.clone())
    
    def detach(self) -> 'RotationTorch':
        '''
        Отсоединяет тензор от графа autograd
        '''
        
        return RotationTorch(self._q.detach())
    
    @property
    def device(self):
        
        return self._q.device
    
    @property
    def dtype(self):
        
        return self._q.dtype
    
    @property
    def shape(self):
        return self._q.shape[:-1] # возвращается только батчева форма размерности, то есть размер кватерниона игнориируется
    
    def __getitem__(self, item):
        return self.q[..., item, :]
    
        
        
        
        
    
    