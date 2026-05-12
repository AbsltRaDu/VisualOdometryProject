from dataclasses import dataclass
from typing import Optional

import torch



class IMUPreprocessor:
    
    def __init__(self, use_lowpass: bool = False, alpha: float = 0.8):
        
        self.use_lowpass = use_lowpass
        self.alpha = alpha
        
    def __call__(self, imu: torch.Tensor, dt: float, state=None):
        '''
        Предоработка IMU
        
        state: Состояние INS для вычитания bias
        dt: Шаг между кадрами
        frame_df: Время между кадрами
        '''
        
        if imu.ndim != 3 or imu.shape[-1] != 6:
            raise ValueError(f"imu должен иметь shape (B, K, 6), а получили {imu.shape}")
        
        
        gyro = imu[..., :3]
        accel = imu[..., 3:]

        if state is not None:
            gyro - gyro - state.b_g[:, None, :] # Вычитаем биасы
            accel = accel - state.b_a[:, None, :]
            
        if self.use_lowpass:
            accel = self._lowpass_sequence(accel)
            
        return {'gyro': gyro, 'accel': accel, 'dt': dt}
            
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
    
    
class PureINSModel:
    '''
    Сухая модель инерциальной одометрии
    '''
    