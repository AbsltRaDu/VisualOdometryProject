import torch
import pytorch3d.transforms as torch3d

from dataclasses import dataclass

from src.geometry.RotationTorch import RotationTorch


@dataclass
class PoseTorch:
    '''
    Класс для работы с позой, элементом группы SE(3)
    
    Представляет преобразования вида:
        T = [ R t ]
            [ 0 1 ]
            
    ВАЖНО: Для корректной работы лучше использовать dtype=float64,
    потому что float32 может приводить к потере информации при округлении
    
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
    
    @staticmethod
    def se3_to_pytorch3d(T: torch.Tensor) -> torch.Tensor:
        '''
        Переводит SE(3)-матрицу из стандартной формы:
            [ R t ]
            [ 0 1 ]
        в формат PyTorch3D:
            [ R 0 ]
            [ t 1 ]
        '''
        
        T_p3d = torch.zeros_like(T)
        T_p3d[..., :3, :3] = T[..., :3, :3]
        T_p3d[..., 3, :3] = T[..., :3, 3]
        T_p3d[..., 3, 3] = 1.0
        return T_p3d
        
    @staticmethod
    def se3_from_pytorch3d(T_p3d: torch.Tensor) -> torch.Tensor:
        '''
        Переводит матрицу из формата PyTorch3D:
            [ R 0 ]
            [ t 1 ]
        в стандартную форму:
            [ R t ]
            [ 0 1 ]
        '''
        
        T = torch.zeros_like(T_p3d)
        T[..., :3, :3] = T_p3d[..., :3, :3]
        T[..., :3, 3] = T_p3d[..., 3, :3]
        T[..., 3, 3] = 1
        
        return T
    
    @classmethod
    def from_lie(cls, xi: torch.Tensor, eps: float = 1e-4) -> 'PoseTorch':
        '''
        Создание позы из элемента алгебры Ли
        
        xi: 6-мерный вектор [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z], где
        rho - логарифм трансляции
        phi - логарифм вращения
        '''
        
        if xi.shape[-1] != 6:
            raise ValueError('Элемент алгебры Ли должен иметь размерность (..., 6)')
        
        batch_shape = xi.shape[:-1]
        xi_flat = xi.reshape(-1, 6)
        
        T_flat = torch3d.se3_exp_map(xi_flat, eps=eps) # Экспоненциальное отображение se(3) -> SE(3)
        T_p3d = T_flat.reshape(*batch_shape, 4, 4)
        
        T = cls.se3_from_pytorch3d(T_p3d)
        
        return cls.from_matrix(T)
    
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
    
    def as_lie(self, eps: float = 1e-4, cos_bound: float = 1e-4) -> torch.Tensor:
        '''
        Возвращает позу в виде элемента алгебры Ли se(3)
        '''
        
        T = self.as_matrix()
        T_p3d = self.se3_to_pytorch3d(T)
        
        batch_shape = T_p3d.shape[:-2]
        T_flat = T_p3d.reshape(-1, 4, 4)
        
        xi_flat = torch3d.se3_log_map(T_flat, eps=eps, cos_bound=cos_bound) # Логарифммическое отображение SE(3) -> se(3)
        
        xi = xi_flat.reshape(*batch_shape, 6)
        
        return xi
    
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
    
    @classmethod
    def stack(cls, poses: list['PoseTorch'], dim: int = 0) -> 'PoseTorch':
        '''
        Аналог torch.stack для объектов PoseTorch
        '''
        
        if len(poses) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка PoseTorch')
        
        if not all(isinstance(p, PoseTorch) for p in poses):
            raise TypeError('Все элементы списка должны быть объектами PoseTorch')
        
        R_quats = torch.stack([p.R.as_quat() for p in poses], dim=dim)
        t = torch.stack([p.t for p in poses], dim=dim)
        
        R = poses[0].R.__class__.from_quat(R_quats)
        
        return cls.from_rt(R, t)
    
    @classmethod
    def cat(cls, poses: list['PoseTorch'], dim: int = 0) -> 'PoseTorch':
        '''
        Аналог torch.cat для объектов PoseTorch
        '''
        
        if len(poses) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка PoseTorch')
        
        if not all(isinstance(p, PoseTorch) for p in poses):
            raise TypeError('Все элементы списка должны быть объектами PoseTorch')
        
        R_quats = torch.cat([p.R.as_quat() for p in poses], dim=dim)
        t = torch.cat([p.t for p in poses], dim=dim)
        
        R = poses[0].R.__class__.from_quat(R_quats)
        
        return cls.from_rt(R, t)
        
    @property
    def device(self):
        
        return self.t.device
    
    @property
    def dtype(self):
        return self.t.dtype
    
    def __getitem__(self, item):
        
        R = self.R[item]
        t = self.t[..., item, :]
        
        return PoseTorch.from_rt(R, t)
        
        
    