from collections import defaultdict

import torch

from src.metrics.KITTI_metrics import get_KITTI_metrices, get_KITTI_metrices2D
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

class CNNTrainerStep:
    
    def __init__(self, model, hidden, pred_pose0, localLoss, globalLoss, optimizer, weight_pose, weight_trajectory, window_size, device):
        
        self.model = model
        
        self.hidden = hidden
        self.pred_pose0 = pred_pose0
        
        self.loss_func = localLoss
        self.loss_func_trajectory = globalLoss
        
        self.optimizer = optimizer
        
        self.weight_pose = weight_pose
        self.weight_trajectory = weight_trajectory
        self.window_size = window_size
        
        self.device = device 
        
        self._unsqueeze_before_trajectoryTorch_marker = True # Чтобы просто не переписывать все ф-ии для RNN и IMU
        
        self.init_metrices()
        
    def init_metrices(self):
        self.loss_mean = 0
        self.loss_pose_mean = 0
        self.loss_mean_trajectory = 0
        self.loss_mean_trajectory_metric = 0
        
        self.p_mean_loc = 0
        self.r_mean_loc = 0
        
        self.r_mean, self.p_mean = 0, 0
        self.path_lengh = 0
    
    def __call__(self, out):
        '''
        Инициализация параметров обучения
        '''
        
        self.x_train = out[0].to(self.device) 
        # self.imu = out[1].to(self.device) 
        self.y_train = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        if self._unsqueeze_before_trajectoryTorch_marker:
            pose = pose.unsqueeze(1)
        
        self.pose = PT.from_lie(pose)
        
        
    def get_predict(self):
        '''
        Получение предикта
        '''
        
        # if self.hidden is not None:
        #     self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict = self.model(self.x_train.to(dtype=torch.float32)) # 6D Вектор алгебры Ли 

        self.predict = predict
        # self.hidden = hidden
        
    def take_predict_after_norm(self, predict, y_train):
        self.predict_denorm = predict
        self.y_train_denorm = y_train
    
    def get_localLoss(self, lm_count):
        '''
        Рассчет локальной ошибки
        '''
        
        predict = self.predict
        
        # Блок рассчета ошибки позы
        loss_pose = self.loss_func(predict, self.y_train)
        
        self.loss_pose_mean = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * self.loss_pose_mean
        self.loss_pose = loss_pose
        
        
    def get_globalLossMetrice(self, lm_count_trajectory):
        '''
        Рассчет метрики ошибки глобальной траектории на последовательности
        '''
        predict = self.predict_denorm
        pred_pose0 = self.pred_pose0
        y_train = self.y_train_denorm
        
        # CNN не работает с последовательностями, а TrajectoryTorch ждет объект (B, S, N)
        if self._unsqueeze_before_trajectoryTorch_marker:
            predict = predict.unsqueeze(1)
            y_train = y_train.unsqueeze(1)
        
        if pred_pose0 is None:
            pred_pose0 = self.pose[0]
            
            self.trajectory_fact_metric = TT.from_lie_relative(y_train, pred_pose0)
            self.trajectory_pred_metric = TT.from_lie_relative(predict.detach(), pred_pose0)
        else:
            trajectory_fact_metric = self.trajectory_fact_metric
            trajectory_pred_metric = self.trajectory_pred_metric
            
            self.trajectory_fact_metric = trajectory_fact_metric.extend_lie_relative(y_train)
            self.trajectory_pred_metric = trajectory_pred_metric.extend_lie_relative(predict.detach())

        with torch.no_grad():
            loss_t_metric_global = self.loss_func_trajectory(self.trajectory_pred_metric[:-1], self.trajectory_fact_metric[:-1])
        
        # TODO Нужно ли тут усреднее?
        self.loss_mean_trajectory_metric = 1 / lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory_metric
        self.pred_pose0 = pred_pose0
        
    def get_globalLoss(self, lm_count_trajectory):
        
        fact_pose = self.pose
        predict = self.predict_denorm
        pred_pose0 = self.pred_pose0
        y_train = self.y_train_denorm
        
        if self._unsqueeze_before_trajectoryTorch_marker:
            predict = predict.unsqueeze(1)
            y_train = y_train.unsqueeze(1)

        trajectory_fact = TT.from_lie_relative(y_train, fact_pose[0])
        trajectory = TT.from_lie_relative(predict, pred_pose0)    
        pred_pose0 = trajectory[-1].poses # Сохранили старую позу
        
        loss_t = self.loss_func_trajectory(trajectory[:-1], trajectory_fact[:-1])
        self.loss_mean_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory

        self.pred_pose0 = pred_pose0
        self.loss_t = loss_t
        
    def step_grad(self, step, lm_count, len_train_bar):
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        loss = self.weight_pose  * loss_pose
        
        is_window_end = (step + 1) % self.window_size == 0
        is_last_step = (step + 1) == len_train_bar
        
        if is_window_end or is_last_step:
            self.hidden = None
            self.pred_pose0 = None
            # loss += self.weight_trajectory * loss_translation
        else:
            self.pred_pose0 = self.pred_pose0.detach()
        
        loss.backward() # TODO нужно ли масштабирование?
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_metrices_test(self, step, lm_count, lm_count_trajectory, len_train_bar):
        '''
        Рассчет основных метрик на валидации
        '''
        
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        trajectory_fact_metric = self.trajectory_fact_metric
        trajectory_pred_metric = self.trajectory_pred_metric
        
        is_window_end = (step + 1) % self.window_size == 0
        is_last_step = (step + 1) == len_train_bar
        
        if is_window_end or is_last_step:
            self.hidden = None
            self.pred_pose0 = None
            
            self.path_lengh = trajectory_fact_metric.path_length()

            motion_fact = trajectory_fact_metric.relative_motion()
            motion_pred = trajectory_pred_metric.relative_motion()
            
            r_mean_loc, p_mean_loc = get_KITTI_metrices(motion_fact, motion_pred, self.path_lengh) 
            
            self.p_mean = 1 / lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.p_mean
            self.r_mean = 1 / lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.r_mean

        loss = self.weight_pose * loss_pose + self.weight_trajectory * loss_translation
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean

class RNNTrainerStep(CNNTrainerStep):
    '''
    Объект trainerStep для RNN моделей
    Добавляетcя Truncated BTTP 
    '''
    
    def __init__(self, model, hidden, pred_pose0, localLoss, globalLoss, optimizer, weight_pose, weight_trajectory, window_size, device):
        super().__init__(
            model=model,
            hidden=hidden,
            pred_pose0=pred_pose0,
            localLoss=localLoss,
            globalLoss=globalLoss,
            optimizer=optimizer,
            weight_pose=weight_pose,
            weight_trajectory=weight_trajectory,
            window_size=window_size,
            device=device
        )
        
        self._unsqueeze_before_trajectoryTorch_marker = False # Чтобы просто не переписывать все ф-ии для RNN и IMU
    
    def get_predict(self):
        '''
        Получение предикта
        '''
        
        if self.hidden is not None:
            self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict, hidden = self.model(self.x_train.to(dtype=torch.float32), self.hidden) # 6D Вектор алгебры Ли 

        self.predict = predict
        self.hidden = hidden
    
    def step_grad(self, step, lm_count, len_train_bar):
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        loss = self.weight_pose * loss_pose + self.weight_trajectory * loss_translation
        
        is_window_end = (step + 1) % self.window_size == 0
        is_last_step = (step + 1) == len_train_bar
        
        if is_window_end or is_last_step:
            self.hidden = None
            self.pred_pose0 = None
            
        else:
            self.pred_pose0 = self.pred_pose0.detach()
        
        loss.backward() # TODO нужно ли масштабирование?
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean
        
        self.optimizer.step()
        self.optimizer.zero_grad()

class DeepVOTrainerStep(RNNTrainerStep):
    def get_predict(self):
        '''
        Получение предикта
        '''
        
        if self.hidden is not None:
            self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict, hidden = self.model(self.x_train.to(dtype=torch.float32), self.hidden) # 6D Вектор алгебры Ли 

        self.predict = PT.from_euler(predict).as_lie()
        self.hidden = hidden
        

class RNNIMUTrainerStep(RNNTrainerStep):
    '''
    Объект trainerStep для RNN VO + IMU модель
    '''
    
    
    def __call__(self, out):
        '''
        Инициализация параметров обучения
        '''
        
        self.x_train = out[0].to(self.device) 
        self.imu = out[1].to(self.device) 
        self.y_train = out[2].to(self.device)
        pose = out[3].to(self.device)
        self.pose = PT.from_lie(pose)
        
    def get_predict(self):
        '''
        Получение предикта
        '''
        
        if self.hidden is not None:
            self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict, hidden = self.model(self.x_train.to(dtype=torch.float32), self.imu.to(dtype=torch.float32), self.hidden) # 6D Вектор алгебры Ли 

        self.predict = predict
        self.hidden = hidden


# БЛОК 2D
from src.geometry.Pose2DTorch import Pose2DTorch as PT2D
from src.geometry.Trajectory2DTorch import Trajectory2DTorch as TT2D
        
class CNNTrainerStep2D(CNNTrainerStep):
    
    def __call__(self, out):
        '''
        Инициализация параметров обучения
        '''
        
        self.x_train = out[0].to(self.device) 
        # self.imu = out[1].to(self.device) 
        self.y_train = out[1].to(self.device)
        pose = out[2].to(self.device)
        
        if self._unsqueeze_before_trajectoryTorch_marker:
            pose = pose.unsqueeze(1)
        
        self.pose = PT2D.from_vector(pose)
        
    def get_globalLossMetrice(self, lm_count_trajectory):
        '''
        Рассчет метрики ошибки глобальной траектории на последовательности
        '''
        predict = self.predict_denorm
        pred_pose0 = self.pred_pose0
        y_train = self.y_train
        
        # CNN не работает с последовательностями, а TrajectoryTorch ждет объект (B, S, N)
        if self._unsqueeze_before_trajectoryTorch_marker:
            predict = predict.unsqueeze(1)
            y_train = y_train.unsqueeze(1)
        
        if pred_pose0 is None:
            pred_pose0 = self.pose[0]
            
            self.trajectory_fact_metric = TT2D.from_vectors_relative(y_train, pred_pose0)
            self.trajectory_pred_metric = TT2D.from_vectors_relative(predict.detach(), pred_pose0)
        else:
            trajectory_fact_metric = self.trajectory_fact_metric
            trajectory_pred_metric = self.trajectory_pred_metric
            
            self.trajectory_fact_metric = trajectory_fact_metric.extend_vectors_relative(y_train)
            self.trajectory_pred_metric = trajectory_pred_metric.extend_vectors_relative(predict.detach())

        with torch.no_grad():
            loss_t_metric_global = self.loss_func_trajectory(self.trajectory_pred_metric[:-1], self.trajectory_fact_metric[:-1])
        
        # TODO Нужно ли тут усреднее?
        self.loss_mean_trajectory_metric = 1 / lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory_metric
        
    def get_globalLoss(self, lm_count_trajectory):
        
        fact_pose = self.pose
        predict = self.predict_denorm
        pred_pose0 = self.pred_pose0
        
        if self._unsqueeze_before_trajectoryTorch_marker:
            predict = predict.unsqueeze(1)

        trajectory_fact = TT2D.from_absolute(fact_pose)
        trajectory = TT2D.from_vectors_relative(predict, pred_pose0)    
        pred_pose0 = trajectory[-1].poses # Сохранили старую позу
        
        loss_t = self.loss_func_trajectory(trajectory[:-1], trajectory_fact)
        self.loss_mean_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory

        self.pred_pose0 = pred_pose0
        self.loss_t = loss_t

    def get_metrices_test(self, step, lm_count, lm_count_trajectory, len_train_bar):
        '''
        Рассчет основных метрик на валидации
        '''
        
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        trajectory_fact_metric = self.trajectory_fact_metric
        trajectory_pred_metric = self.trajectory_pred_metric
        
        is_window_end = (step + 1) % self.window_size == 0
        is_last_step = (step + 1) == len_train_bar
        
        if is_window_end or is_last_step:
            self.hidden = None
            self.pred_pose0 = None
            
            self.path_lengh = trajectory_fact_metric.path_length()

            motion_fact = trajectory_fact_metric.relative_motion()
            motion_pred = trajectory_pred_metric.relative_motion()
            
            r_mean_loc, p_mean_loc = get_KITTI_metrices2D(motion_fact, motion_pred, self.path_lengh) 
            
            self.p_mean = 1 / lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.p_mean
            self.r_mean = 1 / lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.r_mean

        loss = self.weight_pose * loss_pose + self.weight_trajectory * loss_translation
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean


class RNNTrainerStep2D(CNNTrainerStep2D):
    '''
    Объект trainerStep для RNN моделей
    Добавляетcя Truncated BTTP 
    '''
    
    def __init__(self, model, hidden, pred_pose0, localLoss, globalLoss, optimizer, weight_pose, weight_trajectory, window_size, device):
        super().__init__(
            model=model,
            hidden=hidden,
            pred_pose0=pred_pose0,
            localLoss=localLoss,
            globalLoss=globalLoss,
            optimizer=optimizer,
            weight_pose=weight_pose,
            weight_trajectory=weight_trajectory,
            window_size=window_size,
            device=device
        )
        
        self._unsqueeze_before_trajectoryTorch_marker = False # Чтобы просто не переписывать все ф-ии для RNN и IMU
    
    def get_predict(self):
        '''
        Получение предикта
        '''
        
        if self.hidden is not None:
            self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict, hidden = self.model(self.x_train.to(dtype=torch.float32), self.hidden) # 6D Вектор алгебры Ли 

        self.predict = predict
        self.hidden = hidden
    
    def step_grad(self, step, lm_count, len_train_bar):
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        loss = self.weight_pose * loss_pose + self.weight_trajectory * loss_translation
        
        is_window_end = (step + 1) % self.window_size == 0
        is_last_step = (step + 1) == len_train_bar
        
        if is_window_end or is_last_step:
            self.hidden = None
            self.pred_pose0 = None
            
        else:
            self.pred_pose0 = self.pred_pose0.detach()
        
        loss.backward() # TODO нужно ли масштабирование?
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean
        
        self.optimizer.step()
        self.optimizer.zero_grad()