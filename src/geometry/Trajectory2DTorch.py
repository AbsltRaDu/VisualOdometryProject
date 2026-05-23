from dataclasses import dataclass
from typing import Optional, Sequence

import torch 

from src.geometry.Pose2DTorch import Pose2DTorch
from src.geometry.TrajectoryTorch import TrajectoryTorch




@dataclass
class Trajectory2DTorch:
    '''
    Класс для работы с 2D-траекторией.

    Внутреннее представление:
        poses: Pose2DTorch

    Ожидаемая форма poses.pose:
        (..., N, 3)

    где N — длина траектории.
    '''

    poses: Pose2DTorch

    def __post_init__(self):


        if not isinstance(self.poses, Pose2DTorch):
            raise TypeError('poses должен быть объектом Pose2DTorch')

    @classmethod
    def from_absolute(cls, poses: Pose2DTorch) -> 'Trajectory2DTorch':
        '''
        Создаёт 2D-траекторию из абсолютных 2D-поз.
        '''

        return cls(poses)

    @classmethod
    def from_vectors_absolute(cls, poses: torch.Tensor) -> 'Trajectory2DTorch':
        '''
        Создаёт 2D-траекторию из тензора абсолютных поз [x, y, yaw].
        '''

        return cls(Pose2DTorch.from_vector(poses))

    @classmethod
    def from_matrices_absolute(cls, matrices: torch.Tensor) -> 'Trajectory2DTorch':
        '''
        Создаёт 2D-траекторию из SE(2)-матриц 3x3.
        '''

        return cls(Pose2DTorch.from_matrix(matrices))

    @classmethod
    def from_trajectory3d(cls, trajectory3d: TrajectoryTorch, euler_seq: str = "XYZ") -> 'Trajectory2DTorch':
        '''
        Проецирует TrajectoryTorch в Trajectory2DTorch.
        '''

        if not isinstance(trajectory3d, TrajectoryTorch):
            raise TypeError("trajectory3d должен быть объектом TrajectoryTorch")

        poses2d = Pose2DTorch.from_pose3d(trajectory3d.as_pose(), euler_seq=euler_seq)
        return cls.from_absolute(poses2d)

    @classmethod
    def from_relative(cls, deltas: Pose2DTorch, pose0: Optional[Pose2DTorch] = None,) -> 'Trajectory2DTorch':
        '''
        Создаёт абсолютную 2D-траекторию из последовательности относительных движений.

        deltas:
            Pose2DTorch с формой (..., N, 3), где каждый элемент — [dx, dy, d_yaw].
        '''

        if not isinstance(deltas, Pose2DTorch):
            raise TypeError('deltas должен быть объектом Pose2DTorch')

        # Если есть временная размерность N, она находится перед последней размерностью 3.
        if deltas.pose.ndim >= 2:
            batch_shape = deltas.pose.shape[:-2]
            num_deltas = deltas.pose.shape[-2]
        else:
            batch_shape = ()
            num_deltas = 0

        if pose0 is None:
            pose0 = Pose2DTorch.identity(batch_shape=batch_shape, device=deltas.device, dtype=deltas.dtype)

        poses_list = [pose0]
        current = pose0

        if num_deltas > 0:
            for k in range(num_deltas):
                # deltas[k] берёт k-е относительное движение по временной оси.
                current = current * deltas[k]
                poses_list.append(current)

            poses = Pose2DTorch.stack(poses_list, dim=-2)
        else:
            current = current * deltas
            poses_list.append(current)
            poses = Pose2DTorch.stack(poses_list, dim=0)

        return cls(poses)

    @classmethod
    def from_vectors_relative(cls, deltas: torch.Tensor, pose0: Optional[Pose2DTorch] = None) -> 'Trajectory2DTorch':
        '''
        Создаёт траекторию из относительных 2D-векторов [dx, dy, d_yaw].
        '''

        return cls.from_relative(Pose2DTorch.from_vector(deltas), pose0=pose0)

    @classmethod
    def from_lie3d_relative(cls, xi_3d_seq: torch.Tensor, pose0: Optional[Pose2DTorch] = None) -> 'Trajectory2DTorch':
        '''
        Создаёт 2D-траекторию из 3D Ли-приращений.
        '''

        deltas_2d = Pose2DTorch.project_lie3d_to_delta2d(xi_3d_seq)
        return cls.from_vectors_relative(deltas_2d, pose0=pose0)

    def extend_relative(self, deltas: Pose2DTorch, keep_history: bool = True) -> 'Trajectory2DTorch':
        '''
        Продолжает накопленную траекторию, добавляя относительные 2D-движения.
        '''

        if not isinstance(deltas, Pose2DTorch):
            raise TypeError('deltas должен быть Pose2DTorch')

        last_pose = self.last_pose()
        trajectory = self.from_relative(deltas, pose0=last_pose)

        if keep_history:
            return Trajectory2DTorch.cat([Trajectory2DTorch.from_absolute(self.poses[:-1]), trajectory], dim=-2)

        return trajectory

    def extend_vectors_relative(self, deltas: torch.Tensor, keep_history: bool = True) -> 'Trajectory2DTorch':
        '''
        Продолжает траекторию относительными векторами [dx, dy, d_yaw].
        '''

        return self.extend_relative(Pose2DTorch.from_vector(deltas), keep_history=keep_history)

    # ---------------------------------------------------------------------
    # Представления и конвертация
    # ---------------------------------------------------------------------

    def as_pose(self) -> Pose2DTorch:
        '''
        Возвращает траекторию как Pose2DTorch.
        '''

        return self.poses.clone()

    def as_vectors(self) -> torch.Tensor:
        '''
        Возвращает траекторию как тензор [x, y, yaw].
        '''

        return self.poses.as_vector()

    def as_matrices(self) -> torch.Tensor:
        '''
        Возвращает траекторию как SE(2)-матрицы 3x3.
        '''

        return self.poses.as_matrix()

    def to_trajectory3d(self, z: float | torch.Tensor = 0.0, euler_seq: str = 'XYZ') -> TrajectoryTorch:
        '''
        Конвертирует Trajectory2DTorch в TrajectoryTorch.
        '''

        poses3d = self.poses.to_pose3d(z=z, euler_seq=euler_seq)
        return TrajectoryTorch.from_absolute(poses3d)

    def positions(self) -> torch.Tensor:
        '''
        Возвращает XY-координаты траектории.
        '''

        return self.poses.translation()

    def yaws(self) -> torch.Tensor:
        '''
        Возвращает yaw для каждой позы траектории.
        '''

        return self.poses.yaw()

    def relative_deltas(self) -> Pose2DTorch:
        '''
        Вычисляет последовательность относительных движений между соседними позами.
        '''

        deltas = [pose0.inv() * pose1 for pose0, pose1 in zip(self.poses[:-1], self.poses[1:])]
        return Pose2DTorch.stack(deltas, dim=-2)

    def relative_motion(self, start_index: int = 0, end_index: int = -1) -> Pose2DTorch:
        '''
        Возвращает относительное движение от start_index к end_index.
        '''

        return self.poses[start_index].inv() * self.poses[end_index]

    def path_length(self) -> torch.Tensor:
        '''
        Вычисляет длину 2D-траектории по XY-координатам.
        '''

        pos = self.positions()
        diffs = pos[..., 1:, :] - pos[..., :-1, :]
        seq_lengths = torch.linalg.norm(diffs, dim=-1)

        return seq_lengths.sum(dim=-1)

    def first_pose(self) -> Pose2DTorch:
        '''
        Возвращает первую позу траектории.
        '''

        return self.poses[0]

    def last_pose(self) -> Pose2DTorch:
        '''
        Возвращает последнюю позу траектории.
        '''

        return self.poses[-1]

    def clone(self) -> 'Trajectory2DTorch':
        '''
        Создаёт глубокую копию траектории.
        '''

        return Trajectory2DTorch(self.poses.clone())

    def detach(self) -> 'Trajectory2DTorch':
        '''
        Отсоединяет траекторию от графа autograd.
        '''

        return Trajectory2DTorch(self.poses.detach())

    def to(self, *args, **kwargs) -> 'Trajectory2DTorch':
        '''
        Переносит траекторию на другой device или dtype.
        '''

        return Trajectory2DTorch(self.poses.to(*args, **kwargs))

    @classmethod
    def stack(cls, trajectories: list['Trajectory2DTorch'], dim: int = 0) -> 'Trajectory2DTorch':
        '''
        Аналог torch.stack для списка Trajectory2DTorch.
        '''

        if len(trajectories) == 0:
            raise ValueError('Нельзя выполнять stack для пустого списка Trajectory2DTorch')

        if not all(isinstance(t, Trajectory2DTorch) for t in trajectories):
            raise TypeError('Все элементы списка должны быть Trajectory2DTorch')

        return cls(Pose2DTorch.stack([traj.poses for traj in trajectories], dim=dim))

    @classmethod
    def cat(cls, trajectories: list['Trajectory2DTorch'], dim: int = 0) -> 'Trajectory2DTorch':
        '''
        Аналог torch.cat для списка Trajectory2DTorch.
        '''

        if len(trajectories) == 0:
            raise ValueError('Нельзя выполнять cat для пустого списка Trajectory2DTorch')

        if not all(isinstance(t, Trajectory2DTorch) for t in trajectories):
            raise TypeError('Все элементы списка должны быть Trajectory2DTorch')

        return cls(Pose2DTorch.cat([traj.poses for traj in trajectories], dim=dim))

    @property
    def device(self):
        return self.poses.device

    @property
    def dtype(self):
        return self.poses.dtype

    @property
    def shape(self):
        '''
        Возвращает batch-форму траектории без размерностей (N, 3).
        '''

        return self.poses.pose.shape[:-2]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        '''
        Индексация по временной размерности.
        '''

        if self.poses.pose.ndim >= 2:
            pose = self.poses[item]
            return Trajectory2DTorch.from_absolute(pose)

        raise ValueError("Данный объект Trajectory2DTorch не итерируемый")