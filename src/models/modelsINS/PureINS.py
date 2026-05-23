from dataclasses import dataclass
from typing import Optional

import torch

from src.geometry.PoseTorch import PoseTorch
from src.geometry.RotationTorch import RotationTorch




class IMUPreprocessor:
    
    def __init__(self, use_lowpass: bool = False, alpha: float = 0.8):
        
        self.use_lowpass = use_lowpass
        self.alpha = alpha
        
    def __call__(self, imu: torch.Tensor, dt: torch.Tensor, b_g: torch.Tensor, b_a: torch.Tensor):
        '''
        Предоработка IMU
        
        state: Состояние INS для вычитания bias
        dt: Шаг между кадрами
        frame_df: Время между кадрами
        '''
        
        if imu.ndim != 3 or imu.shape[-1] != 6:
            raise ValueError(f'imu должен иметь shape (B, K, 6), а получили {imu.shape}')
        
        
        gyro = imu[..., :3]
        accel = imu[..., 3:]

        gyro = gyro - b_g[..., None, :] # Вычитаем биасы
        accel = accel - b_a[..., None, :]
            
        if self.use_lowpass:
            gyro = self._lowpass_sequence(gyro)
            accel = self._lowpass_sequence(accel)
            
        return gyro, accel, dt
            
    def _lowpass_sequence(self, x: torch.Tensor):
        y = []

        # Первое значение оставляем как есть
        prev = x[:, 0]

        # Добавляем первое значение
        y.append(prev)

        # Идем по временной оси
        for i in range(1, x.shape[1]):
            # Экспоненциальное сглаживание
            prev = self.alpha * prev + (1.0 - self.alpha) * x[:, i]

            # Сохраняем сглаженное значение
            y.append(prev)

        # Собираем обратно в tensor shape (B, K, 3)
        return torch.stack(y, dim=1)


class INSPropagator:
    '''
    Интегратор IMU-данных
    '''
    
    def __init__(self, gravity: Optional[torch.Tensor] = None, use_updated_rotation_for_accel=True):

        self.gravity = gravity
        self.use_updated_rotation_for_accel = use_updated_rotation_for_accel

    def propagate_sequence(self, pose: PoseTorch, velocity: torch.Tensor, gyro: torch.Tensor, accel: torch.Tensor, dt: torch.Tensor, return_all: bool = True):
        '''
        Интегрирует последовательность IMU.

        pose: начальная поза
        velocity: начальная скорость
        gyro: Угловые скорости после предобработки, shape (B, S, 3)
        accel: Ускорения после предобработки, shape (B, S, 3)
        dt: Шаги времени, shape (B, S, 1)

        return_all: Возвращаем все промежуточные состояния
        '''

        # Список для хранения траектории состояний
        states = [] if return_all else None

        # Длина последовательности IMU
        S = gyro.shape[1]

        pose_new = pose
        # Последовательно интегрируем IMU
        for s in range(S):
            # Берем измерение гироскопа на текущем шаге
            gyro_s = gyro[:, s, :]

            # Берем измерение акселерометра на текущем шаге
            accel_s = accel[:, s, :]

            # Берем dt текущего шага и приводим к shape (B, 1)
            dt_s = dt[:, s, :]

            # Выполняем один шаг propagation
            pose_new, velocity = self.step(pose_new, velocity, gyro_s, accel_s, dt_s)

            y = pose.inv() * pose_new
            
            # Сохраняем состояние, если нужно вернуть всю траекторию
            if return_all:
                states.append((pose_new.clone(), velocity.clone()))

        return y, velocity, states

    def step(self, pose: PoseTorch, velocity: torch.Tensor, gyro: torch.Tensor, accel: torch.Tensor, dt: torch.Tensor):
        '''
        Один шаг интегрирования INS
        
        gyro: Угловая скорость в body frame, shape (B, 3), рад/с
        accel: Specific force / ускорение IMU в body frame, shape (B, 3), м/с^2
        dt: Шаг времени, shape (B, 1), сек
        '''

        # Запоминаем старую ориентацию
        R_old = pose.rotation()
        p_old = pose.translation()

        # Интегрируем гироскоп
        delta_rotvec = gyro * dt
        dR = RotationTorch.from_rotvec(delta_rotvec)
        R_new = R_old * dR

        if self.use_updated_rotation_for_accel:
            R_for_accel = R_new # ориентация после шага
        else:
            R_for_accel = R_old # ориентация до шага
        
        f_world = R_for_accel.apply(accel) # выполнил вращение точек

        # Получаем гравитацию нужной формы
        g = self._get_gravity_like(f_world)

        a_world = f_world - g # (B, 3) * (B, 1)
        p_new = p_old + velocity * dt + 0.5 * a_world * dt ** 2 # (B, 1) * (B, 1) и (B, 3) * (B, 1)
        velocity_new = velocity + a_world * dt

        pose_new = PoseTorch.from_rt(R=R_new, t=p_new)
        

        self.debug = {
            "gyro": gyro.detach(),
            "accel_body": accel.detach(),
            "f_world": f_world.detach(),
            "g": g.detach(),
            "a_world": a_world.detach(),
            "dt": dt.detach(),
            "velocity_new": velocity_new.detach(),
            "p_new": p_new.detach(),
        }
        
        
        # Возвращаем новое состояние
        return pose_new, velocity_new

    def _get_gravity_like(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Возвращает gravity tensor с shape (B, 3)
        x: Tensor shape (B, 3).
        '''

        # Если gravity не задана, используем стандартное соглашение ENU-like:
        # z вверх, гравитация вниз.
        if self.gravity is None:
            g = torch.tensor([0.0, 0.0, -9.81], device=x.device, dtype=x.dtype)
        else:
            g = self.gravity.to(device=x.device, dtype=x.dtype)

        # Расширяем gravity до batch shape (B, 3)
        return g.reshape(1, 3).expand(x.shape[0], 3)


class PureINSModel(torch.nn.Module):
    '''
    Сухая модель инерциальной одометрии.
    '''

    def __init__(self, preprocessor: IMUPreprocessor, propagator: INSPropagator):
        '''
        preprocessor: Объект IMUPreprocessor.
        propagator: Объект INSPropagator.
        '''

        # Инициализируем базовый класс PyTorch
        super().__init__()
        
        self.preprocessor = preprocessor # Сохраняем предобработчик IMU
        self.propagator = propagator # Сохраняем интегратор IMU

    def forward(self, imu: torch.Tensor, dt: torch.Tensor, init_pose: PoseTorch, init_velocity: torch.Tensor, b_g: torch.Tensor, b_a: torch.Tensor, return_all: bool = False):
        '''
        Прямой проход Pure INS.

        imu: (B, S, 6)
        dt: Для каждого IMU-измерения свой шаг времени.
        init_pose: Начальная поза PoseTorch.
        init_velocity: Начальная скорость в world frame, shape (B, 3).
        b_g: Bias гироскопа, shape (B, 3).
        b_a: Bias акселерометра, shape (B, 3).
        return_all: Возвращаем промежуточные состояния.
        '''

        # Проверяем формы входных данных
        self._check_inputs(imu=imu, dt=dt, init_velocity=init_velocity, b_g=b_g, b_a=b_a,)

        # Предобрабатываем IMU
        gyro, accel, dt = self.preprocessor(imu=imu, dt=dt, b_g=b_g, b_a=b_a)

        # Интегрируем последовательность IMU
        pose, velocity, states = self.propagator.propagate_sequence(pose=init_pose, velocity=init_velocity, gyro=gyro, accel=accel, dt=dt, return_all=return_all)


        return pose, velocity, states
    

    def _check_inputs(self, imu: torch.Tensor, dt: torch.Tensor, init_velocity: torch.Tensor, b_g: torch.Tensor, b_a: torch.Tensor):
        '''
        Проверяет размерности входных тензоров.
        '''

        # Проверяем IMU: ожидаем (B, S, 6)
        if imu.ndim != 3 or imu.shape[-1] != 6:
            raise ValueError(f'imu должен иметь shape (B, S, 6), получил {imu.shape}')

        # Достаем batch size и длину последовательности
        B, S, _ = imu.shape

        # Проверяем dt: ожидаем (B, S, 1)
        if dt.ndim != 3 or dt.shape != (B, S, 1):
            raise ValueError(f'dt должен иметь shape {(B, S, 1)}, получил {dt.shape}')

        # Проверяем начальную скорость: ожидаем (B, 3)
        if init_velocity.ndim != 2 or init_velocity.shape != (B, 3):
            raise ValueError(f'init_velocity должен иметь shape {(B, 3)}, получил {init_velocity.shape}')

        # Проверяем bias гироскопа: ожидаем (B, 3)
        if b_g.ndim != 2 or b_g.shape != (B, 3):
            raise ValueError(f'b_g должен иметь shape {(B, 3)}, получил {b_g.shape}')

        # Проверяем bias акселерометра: ожидаем (B, 3)
        if b_a.ndim != 2 or b_a.shape != (B, 3):
            raise ValueError(f'b_a должен иметь shape {(B, 3)}, получил {b_a.shape}')