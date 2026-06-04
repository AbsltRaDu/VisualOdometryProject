from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as FilterPyUKF
from filterpy.kalman import MerweScaledSigmaPoints

from src.geometry.PoseTorch import PoseTorch
from src.geometry.RotationTorch import RotationTorch

def to_numpy(x, dtype=np.float64) -> np.ndarray:
    '''
    Переводит torch.Tensor или np.ndarray в NumPy.
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)

    return np.asarray(x, dtype=dtype)

def to_torch(x, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    Приводит входные данные к torch.Tensor.
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)

    return torch.as_tensor(x, device=device, dtype=dtype)

def skew_np(v: np.ndarray) -> np.ndarray:
    """
    Строит кососимметричную матрицу для векторного произведения.

    Эта функция используется внутри NumPy-реализации IMU-прогноза,
    чтобы не создавать torch.Tensor и объекты RotationTorch на каждой
    сигма-точке UKF.
    """
    x, y, z = np.asarray(v, dtype=np.float64)

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    
def so3_exp_np(rotvec: np.ndarray) -> np.ndarray:
    """
    Переводит вектор поворота rotvec в матрицу вращения SO(3).

    rotvec = omega * dt

    Используется формула Родрига. Это NumPy-аналог
    RotationTorch.from_rotvec(...).as_matrix(), но без накладных расходов
    на создание torch.Tensor внутри fx_imu.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    K = skew_np(rotvec)

    if theta < 1e-12:
        # Для очень маленького угла достаточно первого приближения.
        return np.eye(3, dtype=np.float64) + K

    return (
        np.eye(3, dtype=np.float64)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K)
    )

def axis_rotation_np(axis: str, angle: float) -> np.ndarray:
    """
    Возвращает матрицу элементарного поворота вокруг одной оси.
    """
    c = float(np.cos(angle))
    s = float(np.sin(angle))

    if axis == "X":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )

    if axis == "Y":
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

    if axis == "Z":
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    raise ValueError(f"Неизвестная ось вращения: {axis}")


def euler_to_rotmat_np(angles: np.ndarray, seq: str = "XYZ") -> np.ndarray:
    """
    Переводит углы Эйлера в матрицу вращения через NumPy.

    Соглашение соответствует общему принципу PyTorch3D:
    для seq='XYZ' строится произведение Rx @ Ry @ Rz.
    """
    angles = np.asarray(angles, dtype=np.float64).reshape(3)
    seq = seq.upper()

    if len(seq) != 3:
        raise ValueError(f"euler seq должен иметь длину 3, получено: {seq}")

    R = np.eye(3, dtype=np.float64)

    for axis, angle in zip(seq, angles):
        # Последовательно домножаем матрицы в порядке осей seq.
        R = R @ axis_rotation_np(axis, float(angle))

    return R


def rotmat_to_euler_np(R: np.ndarray, seq: str = "XYZ") -> np.ndarray:
    """
    Переводит матрицу вращения в углы Эйлера через NumPy.

    Для скорости и простоты здесь реализованы основные варианты,
    которые обычно используются в проекте: XYZ и ZYX.
    Если нужна другая конвенция, её лучше отдельно добавить формулами.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    seq = seq.upper()

    if seq == "XYZ":
        # Для R = Rx(x) @ Ry(y) @ Rz(z):
        # R[0, 2] = sin(y)
        sy = np.clip(R[0, 2], -1.0, 1.0)
        y = np.arcsin(sy)
        cy = np.cos(y)

        if abs(cy) > 1e-8:
            x = np.arctan2(-R[1, 2], R[2, 2])
            z = np.arctan2(-R[0, 1], R[0, 0])
        else:
            # Сингулярный случай: одну степень свободы фиксируем нулём.
            x = np.arctan2(R[2, 1], R[1, 1])
            z = 0.0

        return wrap_angle_np(np.array([x, y, z], dtype=np.float64))

    if seq == "ZYX":
        # Для R = Rz(z) @ Ry(y) @ Rx(x):
        # Здесь возвращаем углы в порядке seq: [z, y, x].
        sy = np.clip(-R[2, 0], -1.0, 1.0)
        y = np.arcsin(sy)
        cy = np.cos(y)

        if abs(cy) > 1e-8:
            z = np.arctan2(R[1, 0], R[0, 0])
            x = np.arctan2(R[2, 1], R[2, 2])
        else:
            z = np.arctan2(-R[0, 1], R[1, 1])
            x = 0.0

        return wrap_angle_np(np.array([z, y, x], dtype=np.float64))

    raise NotImplementedError(
        f"rotmat_to_euler_np сейчас поддерживает только 'XYZ' и 'ZYX', получено: {seq}"
    )

def wrap_angle_np(a: np.ndarray) -> np.ndarray:
    '''
    Нормализует угол или массив углов в диапазон [-pi, pi].
    '''
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def vo_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    '''
    Вычисляет среднее VO-измерения.

    VO-измерение: z = [x, y, z, roll, pitch, yaw]
    '''
    z = np.dot(Wm, sigmas)

    for angle_idx in (3, 4, 5):
        s = np.dot(Wm, np.sin(sigmas[:, angle_idx]))
        c = np.dot(Wm, np.cos(sigmas[:, angle_idx]))
        z[angle_idx] = np.arctan2(s, c)

    z[3:6] = wrap_angle_np(z[3:6])

    return z

def vo_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Вычисляет остаток VO-измерения.

    Для позиции используется обычная разность.
    Для углов используется нормализация в диапазон [-pi, pi].
    '''
    y = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    y[3:6] = wrap_angle_np(y[3:6])

    return y


def hx_gps(x: np.ndarray) -> np.ndarray:
    '''
    Модель измерения СНС/GPS.

    Из полного состояния фильтра возвращаются только координаты:
        h_gps(x) = [px, py, pz]
    '''
    return x[0:3].copy()


def hx_vo(x: np.ndarray) -> np.ndarray:
    '''
    Модель измерения визуальной одометрии.

    Из полного состояния фильтра возвращаются координаты и ориентация:
        h_vo(x) = [px, py, pz, roll, pitch, yaw]
    '''
    z = np.zeros(6, dtype=np.float64)
    z[0:3] = x[0:3]
    z[3:6] = wrap_angle_np(x[6:9])

    return z


def fx_identity(x: np.ndarray, dt: float) -> np.ndarray:
    '''
    Техническая модель без движения.

    Она нужна для пересчёта сигма-точек перед update, когда несколько
    измерений приходят подряд без нового predict.
    '''
    y = np.asarray(x, dtype=np.float64).copy()
    y[6:9] = wrap_angle_np(y[6:9])

    return y

def state_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    '''
    Вычисляет среднее состояние по сигма-точкам.

    Состояние фильтра: x = [p(3), v(3), rpy(3), ba(3), bg(3)]
    '''
    x = np.dot(Wm, sigmas)

    for angle_idx in (6, 7, 8):
        s = np.dot(Wm, np.sin(sigmas[:, angle_idx]))
        c = np.dot(Wm, np.cos(sigmas[:, angle_idx]))
        x[angle_idx] = np.arctan2(s, c)

    x[6:9] = wrap_angle_np(x[6:9])

    return x

def state_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Вычисляет остаток между двумя состояниями.

    Для линейных компонент используется обычное вычитание.
    Для углов используется нормализация в диапазон [-pi, pi].
    '''
    y = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    y[6:9] = wrap_angle_np(y[6:9])

    return y

def gps_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    '''
    Вычисляет среднее GPS/СНС-измерения [x, y, z].
    '''
    return np.dot(Wm, sigmas)

def gps_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Вычисляет остаток GPS/СНС-измерения [x, y, z].
    '''
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)

@dataclass
class UKFNoiseConfig:
    '''
    Конфигурация ковариаций фильтра
    '''
    
    # Начальная неопределённость состояния.
    p0_pos: float = 5.0
    p0_vel: float = 1.0
    p0_angle: float = 0.2
    p0_accel_bias: float = 0.05
    p0_gyro_bias: float = 0.01

    # Шум процесса, то есть недоверие к инерциальному прогнозу.
    q_pos: float = 0.2
    q_vel: float = 0.2
    q_angle: float = 0.02
    q_accel_bias: float = 1e-4
    q_gyro_bias: float = 1e-5

    # Шум GPS/СНС в метрах.
    r_gps_xyz: Tuple[float, float, float] = (2.0, 2.0, 3.0)

    # Шум VO: позиция в метрах, углы в радианах.
    r_vo_pos: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    r_vo_rpy: Tuple[float, float, float] = (0.1, 0.1, 0.1)

class SigmaPointInertialFusionFilter:
    
    
    def __init__(
        self,
        init_pose: PoseTorch,
        init_velocity: Optional[torch.Tensor] = None,
        noise: Optional[UKFNoiseConfig] = None,
        gravity_xyz: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        gyro_slice: slice = slice(0, 3),
        accel_slice: slice = slice(3, 6),
        R_pose_imu=None,
        euler_seq: str = "XYZ",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        
        
        self.dim_x = 15 # размерность вектора состония
        self.dim_z = 6 # размерность измерения
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.noise = noise if noise is not None else UKFNoiseConfig()
        
        self.gravity_np = to_numpy(np.asarray(gravity_xyz, dtype=np.float64)).reshape(3)
        self.gravity = to_torch(gravity_xyz, device=self.device, dtype=self.dtype).reshape(3)
        
        self.gyro_slice = gyro_slice
        self.accel_slice = accel_slice
        
        if R_pose_imu is None:
            self.R_pose_imu_np = np.eye(3, dtype=np.float64)
        else:
            self.R_pose_imu_np = to_numpy(R_pose_imu).reshape(3, 3)
        
        self.euler_seq = euler_seq
        
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0, subtract=state_residual) # Создание сигма-точек
        
        self.ukf = FilterPyUKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=0.0, fx=self.fx_imu, hx=hx_vo, points=points, 
                               x_mean_fn=state_mean, z_mean_fn=vo_mean, residual_x=state_residual, residual_z=vo_residual)
        
        # TODO
        self.ukf.x = self.make_initial_state(init_pose=init_pose, init_velocity=init_velocity) # Собрали начальое состояние
        
        self.ukf.P = self.make_initial_covariance() # Сгенерили матрицу ковариации
        self.ukf.Q = self.make_process_noise() # Сгенерили шум модели

        self.R_gps = self.make_gps_noise()
        self.R_vo = self.make_vo_noise()
                
        self.refresh_sigmas() # Позволяет выполнять update даже до первого predict
        
    def make_initial_state(self, init_pose: PoseTorch, init_velocity: torch.Tensor) -> np.ndarray:
        '''
        Собирает начальное состояние фильтра.
        '''
        
        position = init_pose.t
        rotation = init_pose.R.as_euler()
        
        x = np.zeros(self.dim_x, dtype=np.float64)

        x[0:3] = to_numpy(position).reshape(3)
        x[3:6] = to_numpy(init_velocity).reshape(3)
        x[6:9] = wrap_angle_np(to_numpy(rotation).reshape(3))

        # Начальные смещения датчиков считаются нулевыми.
        x[9:12] = 0.0001
        x[12:15] = 0.0001

        return x
    
    def make_initial_covariance(self) -> np.ndarray:
        '''
        Создаёт начальную ковариацию P.

        Чем больше значение на диагонали, тем меньше фильтр доверяет
        начальному состоянию.
        '''
        n = self.noise

        diag = np.array(
            [n.p0_pos] * 3
            + [n.p0_vel] * 3
            + [n.p0_angle] * 3
            + [n.p0_accel_bias] * 3
            + [n.p0_gyro_bias] * 3,
            dtype=np.float64,
        )

        return np.diag(diag**2)
    
    def make_process_noise(self) -> np.ndarray:
        '''
        Создаёт ковариацию шума процесса Q.

        Q показывает, насколько фильтр допускает ошибку инерциального
        прогноза между внешними измерениями VO/СНС.
        '''
        n = self.noise

        diag = np.array(
            [n.q_pos] * 3
            + [n.q_vel] * 3
            + [n.q_angle] * 3
            + [n.q_accel_bias] * 3
            + [n.q_gyro_bias] * 3,
            dtype=np.float64,
        )

        return np.diag(diag**2)
    
    def make_gps_noise(self) -> np.ndarray:
        '''
        Создаёт ковариацию измерения GPS/СНС.
        '''
        return np.diag(np.asarray(self.noise.r_gps_xyz, dtype=np.float64) ** 2)

    def make_vo_noise(self) -> np.ndarray:
        '''
        Создаёт ковариацию измерения визуальной одометрии.
        '''
        diag = np.asarray(
            list(self.noise.r_vo_pos) + list(self.noise.r_vo_rpy),
            dtype=np.float64,
        )

        return np.diag(diag**2)
    
    def refresh_sigmas(self) -> None:
        '''
        Пересчитывает сигма-точки вокруг текущих x и P.

        Это нужно, если в один момент времени идут несколько update подряд:
        например, сначала СНС, затем VO.
        '''
        self.ukf.compute_process_sigmas(dt=0.0, fx=fx_identity)
    
    def state_to_pose(self, x: np.ndarray) -> PoseTorch:
        '''
        Переводит вектор состояния FilterPy в объект PoseTorch.

        Здесь геометрическая часть уже выполняется через RotationTorch/PoseTorch,
        а не через самописные NumPy-функции для матриц поворота.
        '''
        x_torch = to_torch(x, device=self.device, dtype=self.dtype)

        t = x_torch[0:3]
        rpy = x_torch[6:9]

        R = RotationTorch.from_euler(rpy, seq=self.euler_seq)

        return PoseTorch.from_rt(R, t)

    def pose_to_state_rpy(self, pose: PoseTorch) -> np.ndarray:
        '''
        Извлекает из PoseTorch углы Эйлера и переводит их в NumPy.
        '''
        rpy = pose.rotation().as_euler(seq=self.euler_seq)
        rpy_np = to_numpy(rpy).reshape(3)

        return wrap_angle_np(rpy_np)

    
    def fx_imu(self, x: np.ndarray, dt: float, imu_sample: np.ndarray) -> np.ndarray:
        '''
        Нелинейная модель движения по IMU.

        Вход:
            x: текущее состояние FilterPy;
            dt: шаг времени;
            imu_sample: один IMU-вектор.

        По умолчанию формат IMU:
            imu_sample = [gx, gy, gz, ax, ay, az]

        Где:
            gyro  — угловая скорость, рад/с;
            accel — линейное ускорение, м/с^2.
        '''
        x_np = np.asarray(x, dtype=np.float64)
        imu_np = np.asarray(imu_sample, dtype=np.float64)
        dt = float(dt)

        # Достаём компоненты состояния.
        p = x_np[0:3]
        v = x_np[3:6]
        rpy = wrap_angle_np(x_np[6:9])
        ba = x_np[9:12]
        bg = x_np[12:15]

        # Достаём измерения IMU и компенсируем оцениваемые bias.
        gyro_imu = imu_np[self.gyro_slice]
        accel_imu = imu_np[self.accel_slice]

        # Достаем bias из состояния.

        ba_imu = x_np[9:12]
        bg_imu = x_np[12:15]

        # Компенсируем bias в той же системе координат, где записано измерение.
        gyro_imu_corr = gyro_imu - bg_imu
        accel_imu_corr = accel_imu - ba_imu
        
        # Переводим гироскоп и акселерометр из IMU-системы в систему PoseTorch-позы.
        gyro = self.R_pose_imu_np @ gyro_imu_corr
        accel_body = self.R_pose_imu_np @ accel_imu_corr
        
        # Текущая ориентация body -> world в виде матрицы вращения.
        R_world_body = euler_to_rotmat_np(rpy, seq=self.euler_seq)

        # Интегрируем ориентацию:
        # R_new = R_old @ Exp(omega_body * dt)
        R_delta = so3_exp_np(gyro * dt)
        R_new = R_world_body @ R_delta
        rpy_new = rotmat_to_euler_np(R_new, seq=self.euler_seq)

        # Переводим ускорение из body-системы в world-систему.
        accel_world = R_world_body @ accel_body + self.gravity_np
        
        # Интегрируем положение и скорость.
        p_new = p + v * dt + 0.5 * accel_world * dt * dt
        v_new = v + accel_world * dt

        # Собираем новое состояние.
        x_new = x_np.copy()
        x_new[0:3] = p_new
        x_new[3:6] = v_new
        x_new[6:9] = rpy_new

        # Bias считаем постоянными, их дрейф задаётся через Q.
        x_new[9:12] = ba
        x_new[12:15] = bg

        return x_new
    
    def normalize_state(self) -> None:
        '''
        Нормализует углы внутри состояния после predict/update.
        '''
        self.ukf.x[6:9] = wrap_angle_np(self.ukf.x[6:9])
        
    def prepare_imu_sequence(self, imu, imu_dt) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Приводит одиночное IMU-измерение или последовательность IMU
        к единому формату:

            imu_seq.shape = (N, D)
            dt_seq.shape  = (N,)

        Поддерживаются формы:
            imu.shape == (D,)
            imu.shape == (N, D)
        '''
        imu_np = to_numpy(imu)
        dt_np = to_numpy(imu_dt).reshape(-1)

        if imu_np.ndim == 1:
            imu_np = imu_np.reshape(1, -1)

        if imu_np.ndim != 2:
            raise ValueError(
                f"imu должен иметь форму (D,) или (N, D), получено {imu_np.shape}"
            )

        if dt_np.size == 1:
            dt_np = np.repeat(float(dt_np[0]), imu_np.shape[0])

        if dt_np.shape[0] != imu_np.shape[0]:
            raise ValueError(
                "Количество imu_dt должно совпадать с количеством IMU-измерений: "
                f"imu={imu_np.shape[0]}, imu_dt={dt_np.shape[0]}"
            )

        return imu_np, dt_np
    
    def predict_imu(self, imu, imu_dt) -> None:
        '''
        Выполняет predict по одному IMU-измерению или по последовательности IMU.
        '''
        imu_seq, dt_seq = self.prepare_imu_sequence(imu, imu_dt)

        for imu_sample, dt in zip(imu_seq, dt_seq):
            self._stabilize_covariance()
            self.ukf.predict(dt=float(dt), imu_sample=imu_sample)
            self._stabilize_covariance()
            self.normalize_state()
            
    def update_gps(self, gps_xyz, R: Optional[np.ndarray] = None) -> None:
        '''
        Выполняет коррекцию по СНС/GPS.

        gps_xyz должен быть представлен в локальных координатах [x, y, z],
        а не в широте, долготе и высоте.
        '''
        z = to_numpy(gps_xyz).reshape(3)

        # Для GPS переключаем функции среднего и остатка измерения.
        self.ukf.z_mean = gps_mean
        self.ukf.residual_z = gps_residual

        self.refresh_sigmas()
        self._stabilize_covariance()
        self.ukf.update(z, R=self.R_gps if R is None else R, hx=hx_gps)
        self._stabilize_covariance()
        self.normalize_state()
        
    def update_vo(self, vo_pose: PoseTorch, R: Optional[np.ndarray] = None) -> None:
        '''
        Выполняет коррекцию по визуальной одометрии.

        vo_pose должен быть объектом PoseTorch с абсолютной накопленной позой
        в той же системе координат, что и состояние фильтра.
        '''
        if not isinstance(vo_pose, PoseTorch):
            raise TypeError("vo_pose должен быть объектом PoseTorch")

        vo_pose = vo_pose.detach().to(device=self.device, dtype=self.dtype)

        z = np.zeros(6, dtype=np.float64)
        z[0:3] = to_numpy(vo_pose.translation()).reshape(3)
        z[3:6] = self.pose_to_state_rpy(vo_pose)

        # Для VO переключаем функции среднего и остатка измерения.
        self.ukf.z_mean = vo_mean
        self.ukf.residual_z = vo_residual

        self.refresh_sigmas()
        self._stabilize_covariance()
        self.ukf.update(
            z,
            R=self.R_vo if R is None else R,
            hx=hx_vo,
        )
        self._stabilize_covariance()
        self.normalize_state()
        
    def step(self, imu=None, imu_dt=None, vo_pose: Optional[PoseTorch] = None, gps_xyz=None) -> dict:
        '''
        Выполняет полный шаг фильтра.

        Порядок обработки:
            1. predict по IMU;
            2. update по GPS/СНС;
            3. update по VO.
        '''
        if imu is not None:
            if imu_dt is None:
                raise ValueError("Если передан imu, обязательно нужно передать imu_dt")

            self.predict_imu(imu, imu_dt)

        if gps_xyz is not None:
            self.update_gps(gps_xyz)

        if vo_pose is not None:
            self.update_vo(vo_pose)

        return self.get_state_torch()
    
    def get_state_numpy(self) -> dict:
        '''
        Возвращает текущее состояние фильтра в NumPy.
        '''
        x = self.ukf.x.copy()
        x[6:9] = wrap_angle_np(x[6:9])

        return {
            "position": x[0:3].copy(),
            "velocity": x[3:6].copy(),
            "rpy": x[6:9].copy(),
            "accel_bias": x[9:12].copy(),
            "gyro_bias": x[12:15].copy(),
            "covariance": self.ukf.P.copy(),
        }
        
    def get_state_torch(self) -> dict:
        '''
        Возвращает текущее состояние фильтра в torch.Tensor.
        '''
        state = self.get_state_numpy()

        return {
            key: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for key, value in state.items()
        }

    def get_pose(self) -> PoseTorch:
        '''
        Возвращает текущую оценку позы как PoseTorch.
        '''
        return self.state_to_pose(self.ukf.x)

    def get_pose_matrix_torch(self) -> torch.Tensor:
        '''
        Возвращает текущую оценку позы как матрицу SE(3) формы 4x4.
        '''
        return self.get_pose().as_matrix()

    def reset(self, init_position=None, init_velocity=None, init_rpy=None) -> None:
        '''
        Сбрасывает фильтр в новое начальное состояние.
        '''
        self.ukf.x = self.make_initial_state(
            init_position=init_position,
            init_velocity=init_velocity,
            init_rpy=init_rpy,
        )

        self.ukf.P = self.make_initial_covariance()
        self.normalize_state()
        self.refresh_sigmas()
        
    def test_raw_fx_imu(ukf, imu, imu_dt):
        """
        Проверяет чистое инерциальное счисление без UKF, P, Q и сигма-точек.
        Если здесь тоже улетает, проблема в dt, ориентации, гравитации или осях.
        Если здесь нормально, а через ukf.predict() улетает, проблема в сигма-точках/P.
        """
        imu_seq, dt_seq = ukf.prepare_imu_sequence(imu, imu_dt)

        # Берём только центральное состояние фильтра.
        x = ukf.ukf.x.copy()

        for i, (imu_sample, dt) in enumerate(zip(imu_seq, dt_seq)):
            # Прогоняем одну обычную модель движения без UKF.
            x = ukf.fx_imu(x, float(dt), imu_sample)

            print(
                i,
                "dt =", float(dt),
                "p =", x[0:3],
                "v =", x[3:6],
                "rpy =", x[6:9],
            )

        return x   
    
    def _stabilize_covariance(self, eps: float = 1e-8) -> None:
        """
        Численно стабилизирует матрицу ковариации P.

        UKF требует, чтобы P была симметричной и положительно определенной.
        Из-за численных ошибок после predict/update P может стать слегка
        несимметричной или получить маленькие отрицательные собственные значения.
        """

        # Берем текущую ковариацию фильтра
        P = self.ukf.P

        # Проверяем, что в P нет NaN и inf
        if not np.isfinite(P).all():
            raise FloatingPointError("Матрица P содержит NaN или inf")

        # Принудительно симметризуем P
        P = 0.5 * (P + P.T)

        # Считаем собственные значения симметричной P
        eigvals = np.linalg.eigvalsh(P)

        # Находим минимальное собственное значение
        min_eig = eigvals.min()

        # Если P не положительно определена, добавляем jitter на диагональ
        if min_eig < eps:
            P = P + np.eye(P.shape[0], dtype=P.dtype) * (eps - min_eig + eps)

        # Дополнительно защищаем диагональ от нулевых/отрицательных дисперсий
        diag = np.diag(P).copy()
        diag = np.maximum(diag, eps)
        np.fill_diagonal(P, diag)

        # Сохраняем стабилизированную ковариацию обратно в UKF
        self.ukf.P = P