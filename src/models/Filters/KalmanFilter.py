from dataclasses import dataclass
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter



@dataclass
class LooseVOINSFilterConfig:
    '''
    Конфиг слабосвязнного фильтра ВО + ИНС
    
    position_std: Начальная неопределенность положения.
    velocity_std: Начальная неопределенность скорости.
    process_accel_std: Шум модели движения, связанный с ошибками ускорения INS.
    vo_position_std: Шум VO-измерения положения.
    '''
    
    position_std: float = 0.1
    velocity_std: float = 1.0
    process_accel_std: float = 0.5
    vo_position_std: float = 0.3
    

class LooseVOINSLianerKalmanFilter:
    '''
    Простой слабосвязный линейный фильтр ВО + ИНС
    
    Состояние фильтра: [px, py, pz, vx, vy, vz]
    '''
    
    def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, config: Optional[LooseVOINSFilterConfig] = None):
        '''
        init_position: Начальное положение, shape (3,).

        init_velocity: Начальная скорость, shape (3,).

        config: Параметры неопределенности фильтра.
        '''
        
        self.config = config if config else LooseVOINSFilterConfig()
        
        init_position = self._as_vector3(init_position, name="init_position")
        init_velocity = self._as_vector3(init_velocity, name="init_velocity")

        # Инициализация фильтра Калмана
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.zeros((6, 1), dtype=np.float64)
        self.kf.x[0:3, 0] = init_position
        self.kf.x[3:6, 0] = init_velocity

        # Начальная ковариация состояния
        self.kf.P = np.diag(
            [
                self.config.position_std**2,
                self.config.position_std**2,
                self.config.position_std**2,
                self.config.velocity_std**2,
                self.config.velocity_std**2,
                self.config.velocity_std**2,
            ]
        ).astype(np.float64)
        
        # матрица измерения внешним датчиком
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        
        # Ковариация шума измерения (уровень доверия к внешнему датчику)
        self.kf.R = np.eye(3, dtype=np.float64) * (self.config.vo_position_std**2)
        
        # Инициаилизация матрицы перехода и матрицу шума ускорения
        self.kf.F = np.eye(6, dtype=np.float64)
        self.kf.Q = np.eye(6, dtype=np.float64) * 1e-6

    def predict(self, accel_world: np.ndarray, dt: float) -> None:
        '''
        Шаг предсказания по INS.

        accel_world: Ускорение в мировой системе координат, shape (3,).
        dt: Время между шагами, секунды.
        '''

        # Проверяем ускорение и приводим к shape (3,)
        accel_world = self._as_vector3(accel_world, name='accel_world')

        # Проверяем dt
        if dt <= 0.0:
            raise ValueError(f"dt должен быть положительным, получили {dt}")

        # Строим матрицу перехода состояния F:
        F = self._make_transition_matrix(dt)

        # Строим матрицу управления B:
        B = self._make_control_matrix(dt)

        # Строим шум процесса Q.
        Q = self._make_process_noise(dt)

        # Сохраняем F и Q внутри FilterPy-объекта для совместимости
        self.kf.F = F
        self.kf.Q = Q

        # Переводим acceleration в столбец shape (3, 1)
        u = accel_world.reshape(3, 1)

        self.kf.x = F @ self.kf.x + B @ u # Выполняем predict вручную, потому что у нас есть управляющее воздействие B @ u

        self.kf.P = F @ self.kf.P @ F.T + Q # Обновляем ковариацию состояния

    def update_vo_position(self, vo_position: np.ndarray) -> None:
        '''
        Шаг коррекции по визуальной одометрии.

        vo_position: Абсолютная позиция, восстановленная из VO, shape (3,).
        '''

        z = self._as_vector3(vo_position, name="vo_position").reshape(3, 1)

        
        self.kf.update(z) # Выполняем стандартный update FilterPy

    def get_position(self) -> np.ndarray:
        '''
        Возвращает текущую оценку положения shape (3,).
        '''

        return self.kf.x[0:3, 0].copy()

    def get_velocity(self) -> np.ndarray:
        '''
        Возвращает текущую оценку скорости shape (3,).
        '''

        return self.kf.x[3:6, 0].copy()

    def get_state(self) -> np.ndarray:
        '''
        Возвращает полный state shape (6,).
        '''

        return self.kf.x[:, 0].copy()

    def get_covariance(self) -> np.ndarray:
        '''
        Возвращает текущую ковариацию состояния P shape (6, 6).
        '''

        return self.kf.P.copy()

    def set_vo_position_std(self, vo_position_std: float) -> None:
        '''
        Позволяет динамически менять доверие к VO.

        Меньше vo_position_std -> больше доверяем VO.
        Больше vo_position_std -> меньше доверяем VO.
        '''

        # Обновляем параметр в конфиге
        self.config.vo_position_std = float(vo_position_std)

        # Пересобираем R
        self.kf.R = np.eye(3, dtype=np.float64) * (self.config.vo_position_std**2)

    def _make_translation_matrix(self, dt: float) -> np.ndarray:
        '''
        Создает матрицу переходного состояния
        '''
        
        F = np.eye(6, dtype=np.float64)
        
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        return F
    
    def _make_control_matrix(self, dt: float) -> np.ndarray:
        '''
        Создает матрицу учета ускорения
        '''

        B = np.zeros((6, 3), dtype=np.float64)

        # Ускорение влияет на положение через 0.5 * a * dt^2
        B[0, 0] = 0.5 * dt**2
        B[1, 1] = 0.5 * dt**2
        B[2, 2] = 0.5 * dt**2

        # Ускорение влияет на скорость через a * dt
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        return B
    
    def _make_process_noise(self, dt: float) -> np.ndarray:
        '''
        Создает матрицу шума ускорения
        '''

        # Дисперсия шума ускорения
        sigma2 = self.config.process_accel_std**2

        Q = np.zeros((6, 6), dtype=np.float64)

        # Компоненты шума для модели position/velocity
        q_pos = 0.25 * dt**4 * sigma2
        q_cross = 0.5 * dt**3 * sigma2
        q_vel = dt**2 * sigma2

        # Заполняем одинаковую структуру для X, Y, Z
        for axis in range(3):
            
            p_idx = axis # Индекс положения по текущей оси
            v_idx = axis + 3 # Индекс скорости по текущей оси

            # Шум положения
            Q[p_idx, p_idx] = q_pos

            # Корреляция шума положения и скорости
            Q[p_idx, v_idx] = q_cross
            Q[v_idx, p_idx] = q_cross

            # Шум скорости
            Q[v_idx, v_idx] = q_vel

        return Q
    
    @staticmethod
    def _as_vector3(value: np.ndarray, name: str) -> np.ndarray:
        '''
        Проверяет, что вход можно представить как вектор shape (3,).
        '''

        # Приводим к NumPy-массиву float64
        array = np.asarray(value, dtype=np.float64).reshape(-1)

        # Проверяем количество элементов
        if array.shape[0] != 3:
            raise ValueError(f"{name} должен иметь 3 элемента, получили shape {array.shape}")

        return array