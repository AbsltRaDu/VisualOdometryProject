from dataclasses import dataclass
from typing import Optional, Sequence

import torch 

from src.geometry.PoseTorch import PoseTorch
from src.geometry.RotationTorch import RotationTorch

@dataclass
class Pose2DTorch:
    '''
    Класс для работы с позой в плоскости XY.

    Внутреннее представление:
        pose = [x, y, yaw]

    где:
        x   — координата по оси X;
        y   — координата по оси Y;
        yaw — угол рыскания вокруг оси Z в радианах.
    
    '''
    
    pose: torch.Tensor
    
    def __post_init__(self):
        
        if not isinstance(self.pose, torch.Tensor):
            raise TypeError('pose2D должен быть torch.Tensor')
        
        if self.pose.shape[-1] != 3:
            raise ValueError(f"Pose2DTorch ожидает форму (..., 3), получено {self.pose.shape}")
        
        self.t = self.translation()
        self.R = self.yaw()
        
    @classmethod
    def from_xy_yaw(cls, xy: torch.Tensor, yaw: torch.Tensor) -> "Pose2DTorch":
        '''
        Создаёт Pose2DTorch из трансляции xy и угла yaw.

        '''

        if xy.shape[-1] != 2:
            raise ValueError(f"xy должен иметь форму (..., 2), получено {xy.shape}")

        if yaw.shape[-1:] == (1,):
            yaw = yaw.squeeze(-1)

        pose = torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)

        return cls(pose)
    
    @classmethod
    def from_vector(cls, pose: torch.Tensor) -> "Pose2DTorch":
        '''
        Создаёт Pose2DTorch из готового вектора [x, y, yaw].
        '''

        return cls(pose)
    
    @classmethod
    def identity(cls, batch_shape: Sequence[int] | torch.Size = (), device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> 'Pose2DTorch':
        '''
        Создаёт единичную 2D-позу.

        Единичная поза:
            x = 0
            y = 0
            yaw = 0
        '''

        pose = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
        return cls(pose)
    
    @classmethod
    def from_matrix(cls, T: torch.Tensor) -> 'Pose2DTorch':
        '''
        Создаёт Pose2DTorch из SE(2)-матрицы 3x3.

        Матрица вида:
            [ cos(yaw)  -sin(yaw)   x ]
            [ sin(yaw)   cos(yaw)   y ]
            [    0          0       1 ]
        '''

        if T.shape[-2:] != (3, 3):
            raise ValueError(f"SE(2)-матрица должна иметь форму (..., 3, 3), получено {T.shape}")

        x = T[..., 0, 2]
        y = T[..., 1, 2]

        # Угол извлекается из матрицы поворота SO(2).
        yaw = torch.atan2(T[..., 1, 0], T[..., 0, 0])

        return cls(torch.stack([x, y, yaw], dim=-1))
    
    @classmethod
    def from_pose3d(cls, pose3d: PoseTorch, euler_seq: str = "XYZ") -> 'Pose2DTorch':
        '''
        Проецирует 3D-позу PoseTorch в 2D-позу.

        Берём:
            x   = t_x
            y   = t_y
            yaw = угол вокруг Z

        Отбрасываем:
            z, roll, pitch

        '''

        if not isinstance(pose3d, PoseTorch):
            raise TypeError("pose3d должен быть объектом PoseTorch")

        # Забираем x, y из 3D-трансляции.
        xy = pose3d.t[..., :2]

        # Получаем углы Эйлера и берём yaw.
        # Для convention='XYZ': angles = [roll_x, pitch_y, yaw_z].
        euler = pose3d.R.as_euler(seq=euler_seq)
        yaw = euler[..., 2]

        return cls.from_xy_yaw(xy, yaw)
    
    def to_pose3d(self, z: float | torch.Tensor = 0.0, euler_seq: str = "XYZ") -> PoseTorch:
        '''
        Конвертирует 2D-позу в 3D-позу PoseTorch.

        Делает ограниченную SE(3) позу:
            x = x
            y = y
            yaw = yaw

        При этом:
        z = const
        roll = 0
        pitch = 0
        '''

        xy_yaw = self.as_vector()
        batch_shape = xy_yaw.shape[:-1]

        t = torch.zeros(*batch_shape, 3, device=self.device, dtype=self.dtype)
        t[..., 0] = xy_yaw[..., 0]
        t[..., 1] = xy_yaw[..., 1]

        # z может быть числом или тензором.
        if isinstance(z, torch.Tensor):
            t[..., 2] = z.to(device=self.device, dtype=self.dtype)
        else:
            t[..., 2] = float(z)

        angles = torch.zeros(*batch_shape, 3, device=self.device, dtype=self.dtype)
        angles[..., 2] = xy_yaw[..., 2]

        R = RotationTorch.from_euler(angles, seq=euler_seq)

        return PoseTorch.from_rt(R, t)
    
    @classmethod
    def project_lie3d_to_delta2d(cls, xi_3d: torch.Tensor) -> 'Pose2DTorch':
        '''
        Проецирует 6D Ли-вектор SE(3) в 2D-приращение [dx, dy, d_yaw].

        Ожидаемый порядок xi_3d:
            [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]

        где:
            rho_* — трансляционная часть;
            phi_* — rotvec/ось-угол часть.

        Возвращает:
            [x, y, yaw]
        '''

        if xi_3d.shape[-1] != 6:
            raise ValueError(f"xi_3d должен иметь форму (..., 6), получено {xi_3d.shape}")

        pose3d = PoseTorch.from_lie(xi_3d)
        
        return cls.from_pose3d(pose3d)
    
    def to_lie3d(self) -> PoseTorch:
        '''
        Конвентирует объект Pose2DTorch В Ли алгебру PoseTorch
        
        Возвращает:
            [dx, dy, 0, 0, 0, d_yaw]
        '''
        
        pose3d = self.to_pose3d()
        return pose3d.as_lie()
        
    @classmethod
    def project_delta2d_to_lie3d(cls, delta_2d: torch.Tensor) -> torch.Tensor:
        '''
        Поднимает 2D-приращение [dx, dy, d_yaw] в 6D Lie-вектор SE(3).

        Возвращает:
            [dx, dy, 0, 0, 0, d_yaw]

        '''

        if delta_2d.shape[-1] != 3:
            raise ValueError(f"delta_2d должен иметь форму (..., 3), получено {delta_2d.shape}")

        pose2d = cls.from_vector(delta_2d)

        return pose2d.to_lie3d()
    
    def as_vector(self) -> torch.Tensor:
        '''
        Возвращает позу как [x, y, yaw].
        '''

        return self.pose.clone()
    
    def as_matrix(self) -> torch.Tensor:
        '''
        Возвращает позу как SE(2)-матрицу 3x3.
        '''

        x = self.pose[..., 0]
        y = self.pose[..., 1]
        yaw = self.pose[..., 2]

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        batch_shape = self.pose.shape[:-1]
        T = torch.zeros(*batch_shape, 3, 3, device=self.device, dtype=self.dtype)

        T[..., 0, 0] = c
        T[..., 0, 1] = -s
        T[..., 1, 0] = s
        T[..., 1, 1] = c
        T[..., 0, 2] = x
        T[..., 1, 2] = y
        T[..., 2, 2] = 1.0

        return T
    
    def translation(self) -> torch.Tensor:
        '''
        Возвращает 2D-трансляцию [x, y].
        '''

        return self.pose[..., :2].clone()

    def yaw(self) -> torch.Tensor:
        '''
        Возвращает угол yaw.
        '''

        return self.pose[..., 2].clone()
    
    def compose(self, other: 'Pose2DTorch') -> 'Pose2DTorch':
        '''
        Композиция двух 2D-поз.
        '''

        if not isinstance(other, Pose2DTorch):
            raise TypeError("other должен быть объектом Pose2DTorch")

        x = self.pose[..., 0]
        y = self.pose[..., 1]
        yaw = self.pose[..., 2]

        dx = other.pose[..., 0]
        dy = other.pose[..., 1]
        d_yaw = other.pose[..., 2]

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x_new = x + c * dx - s * dy
        y_new = y + s * dx + c * dy
        yaw_new = yaw + d_yaw

        return Pose2DTorch(torch.stack([x_new, y_new, yaw_new], dim=-1))
    
    def inv(self) -> 'Pose2DTorch':
        '''
        Возвращает обратную 2D-позу.

        Если:
            p_global = R * p_local + t

        то обратное преобразование:
            p_local = R^T * (p_global - t)
        '''

        x = self.pose[..., 0]
        y = self.pose[..., 1]
        yaw = self.pose[..., 2]

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        # t_inv = -R^T @ t
        x_inv = -(c * x + s * y)
        y_inv = -(-s * x + c * y)
        yaw_inv = -yaw

        return Pose2DTorch(torch.stack([x_inv, y_inv, yaw_inv], dim=-1))

    def between(self, other: "Pose2DTorch") -> "Pose2DTorch":
        '''
        Возвращает относительное движение от self к other:
            delta = self^{-1} * other
        '''

        return self.inv() * other
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        '''
        Применяет 2D-позу к точкам.

        points:
            (..., 2)

        Возвращает:
            (..., 2)
        '''

        if points.shape[-1] != 2:
            raise ValueError(f"points должен иметь форму (..., 2), получено {points.shape}")

        yaw = self.pose[..., 2]
        c = torch.cos(yaw)
        s = torch.sin(yaw)

        px = points[..., 0]
        py = points[..., 1]

        x_new = c * px - s * py + self.pose[..., 0]
        y_new = s * px + c * py + self.pose[..., 1]

        return torch.stack([x_new, y_new], dim=-1)
    
    def __mul__(self, other: 'Pose2DTorch') -> 'Pose2DTorch':

        return self.compose(other)

    def clone(self) -> "Pose2DTorch":
        '''
        Создаёт глубокую копию объекта.
        '''

        return Pose2DTorch(self.pose.clone())
    
    def detach(self) -> 'Pose2DTorch':
        '''
        Отсоединяет pose от графа autograd.
        '''

        return Pose2DTorch(self.pose.detach())

    def to(self, *args, **kwargs) -> 'Pose2DTorch':
        '''
        Переносит pose на другой device или dtype.
        '''

        return Pose2DTorch(self.pose.to(*args, **kwargs))

    @classmethod
    def stack(cls, poses: list['Pose2DTorch'], dim: int = 0) -> 'Pose2DTorch':
        '''
        Аналог torch.stack для списка Pose2DTorch.
        '''

        if len(poses) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка Pose2DTorch')

        if not all(isinstance(p, Pose2DTorch) for p in poses):
            raise TypeError('Все элементы списка должны быть Pose2DTorch')

        return cls(torch.stack([p.pose for p in poses], dim=dim))

    @classmethod
    def cat(cls, poses: list['Pose2DTorch'], dim: int = 0) -> 'Pose2DTorch':
        '''
        Аналог torch.cat для списка Pose2DTorch.
        '''

        if len(poses) == 0:
            raise ValueError('Нельзя выполнять cat для пустого списка Pose2DTorch')

        if not all(isinstance(p, Pose2DTorch) for p in poses):
            raise TypeError('Все элементы списка должны быть Pose2DTorch')

        return cls(torch.cat([p.pose for p in poses], dim=dim))

    @property
    def device(self):
        '''Возвращает device внутреннего тензора.'''

        return self.pose.device

    @property
    def dtype(self):
        '''Возвращает dtype внутреннего тензора.'''

        return self.pose.dtype

    @property
    def shape(self):
        '''
        Возвращает batch-форму без последней размерности 3.
        '''

        return self.pose.shape[:-1]

    def __len__(self):
        '''
        Возвращает длину последовательности, если объект содержит последовательность поз.
        '''

        if self.pose.ndim > 1:
            return self.pose.shape[-2]
        return 1

    def __getitem__(self, item):
        '''
        Индексация по временной размерности.

        Логика согласована с твоим PoseTorch:
            если pose имеет форму (..., N, 3), то item выбирает элемент по N.
        '''

        if self.pose.ndim >= 2:
            return Pose2DTorch(self.pose[..., item, :])

        raise ValueError('Данный объект Pose2DTorch не итерируемый')