import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm

from src.metrics.KITTI_metrics import translation_rmse_drift, rotation_rmse_drift
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

from src.function_of_loss.mse_pose import PoseLossTrajectory, PoseLoss

class trainingProgressiveVIO:
    
    def __init__(self, train_data, test_data, model, loss_func_pose, 
                loss_func_trajectory, optimizer, epochs, device, 
                name_of_model, path_to_save_process_of_fitting,
                normalize=None, best_score = 10**10, weight_pose=1, weight_trajectory=0, window_size=20):
        
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss_func_pose = loss_func_pose
        self.loss_func_trajectory = loss_func_trajectory
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.name_of_model = name_of_model
        self.path_to_save_process_of_fitting = path_to_save_process_of_fitting
        self.normalize = normalize
        self.best_score = best_score
        
        self.weight_trajectory = weight_trajectory
        self.weight_pose  = weight_pose
        self.window_size = window_size
    
    
        # Блок подготовки словаря для записи
        if os.path.isfile(self.path_to_save_process_of_fitting):
            with open(self.path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
                self.dct_of_results = json.load(f)
                
            self._start = self.dct_of_results['epoch'][-1] + 1
            self._end = self._start + epochs
            
            self.best_score = self.dct_of_results['Average Translational RMSE drift'][-1] + self.dct_of_results['Average Rotational RMSE drift'][-1]
            
        else:   
            self.dct_of_results = defaultdict(list)
            self._start = 0
            self._end = epochs
    
    def _init_metrices(self):
        self.loss_mean_train = 0
        self.loss_pose_mean_train = 0
        self.loss_mean_train_trajectory = 0
        self.loss_mean_trajectory_metric = 0
        
        self.loss_mean_test = 0
        self.loss_pose_mean_test = 0
        self.loss_mean_test_trajectory = 0
        self.r_mean, self.p_mean = 0, 0
        self.path_lengh = 0
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
    
    def _train(self, epoch):
        train_bar = tqdm(self.train_data, desc=f'Эпоха тренировочная {epoch+1}/{self._end}', position=0)
            
        self.model.train()
        
        hidden = None
        pred_pose0 = None
        
        for step, (x_train, imu, _, y_train, pose) in enumerate(train_bar):
            x_train = x_train.to(self.device) 
            imu = imu.to(self.device) 
            y_train = y_train.to(self.device)
            pose = pose.to(self.device)

            if hidden is not None:
                hidden = self.model.detach_hidden(hidden) # отрубили от графа вычислений предыдущий скрытый слой
            
            predict, hidden = self.model(x_train.to(dtype=torch.float32), imu.to(dtype=torch.float32), hidden) # 6D Вектор алгебры Ли 
            predict = predict
            
            # Блок рассчета ошибки позы
            loss_pose = self.loss_func_pose(predict, y_train)
            self.lm_count += 1
            self.loss_pose_mean_train = 1 / self.lm_count * loss_pose.item() + (1 - 1 / self.lm_count) * self.loss_pose_mean_train
            
            # блок денормализации
            if self.normalize:
                predict = self.normalize.denormalize(predict)
            
            # Блок рассчета ошибки окна
            self.lm_count_trajectory += 1
            fact_pose = PT.from_lie(pose)
            if pred_pose0 is None:
                pred_pose0 = fact_pose[0]
                trajectory_fact_metric = TT.from_lie_relative(y_train, pred_pose0)
                trajectory_pred_metric = TT.from_lie_relative(predict.detach(), pred_pose0)
            else:
                pred_pose0 = pred_pose0.detach() # Отрубаем предыдущую позу от графа вычислений
                trajectory_fact_metric = trajectory_fact_metric.extend_lie_relative(y_train)
                trajectory_pred_metric = trajectory_pred_metric.extend_lie_relative(predict.detach())
            
            trajectory_fact = TT.from_absolute(fact_pose)
            trajectory = TT.from_lie_relative(predict, pred_pose0)    
            pred_pose0 = trajectory[-2].poses # Сохранили старую позу
            
            loss_t = self.loss_func_trajectory(trajectory[:-1], trajectory_fact)
            self.loss_mean_train_trajectory = 1 / self.lm_count_trajectory * loss_t.item() + (1 - 1 / self.lm_count_trajectory) * self.loss_mean_train_trajectory

            # Чтоб не дай бог градиент не посчитался по всей последовательности
            with torch.no_grad():
                loss_t_metric_global = self.loss_func_trajectory(trajectory_pred_metric[:-1], trajectory_fact_metric[:-1])
            self.loss_mean_trajectory_metric = 1 / self.lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / self.lm_count_trajectory) * self.loss_mean_trajectory_metric
            
            loss = self.weight_pose  * loss_pose
            
            
            if (step + 1) % self.window_size == 0:
                hidden = None
                pred_pose0 = None
                loss += self.weight_trajectory * loss_t
            
            loss.backward() # TODO нужно ли масштабирование?
            self.loss_mean_train = 1 / self.lm_count * loss.item() + (1 - 1 / self.lm_count) * self.loss_mean_train
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            train_bar.set_postfix({
                'loss': self.loss_mean_train,
                'loss_pose': self.loss_pose_mean_train,
                'loss_trajectory_win': self.loss_mean_train_trajectory,
                'loss_trajectory_gl': self.loss_mean_trajectory_metric
            })
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _test(self, epoch):
        
        self.model.eval()
            
        val_bar = tqdm(self.test_data, desc=f'Эпоха валидационная {epoch+1}/{self._end}', position=1)
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
        hidden = None
        pred_pose0 = None 
        loss_t_metric_global = 0 # При инициализации
        
        # TODO доделать оконный метод
        for step, (x_test, imu_test, _, y_val, pose) in enumerate(val_bar):
            x_test = x_test.to(self.device) 
            imu_test = imu_test.to(self.device) 
            y_val = y_val.to(self.device)
            pose = pose.to(self.device)
            
            with torch.no_grad():
                predict, hidden = self.model(x_test.to(dtype=torch.float32), imu_test.to(dtype=torch.float32), hidden)
                predict = predict
                
                loss_pose = self.loss_func_pose(predict, y_val)
                
                if self.normalize:
                    predict = self.normalize.denormalize(predict)

                
                fact_pose = PT.from_lie(pose)
                if pred_pose0 is None:
                    pred_pose0 = fact_pose[0]
                    trajectory_fact_metric = TT.from_lie_relative(y_val, pred_pose0)
                    trajectory_pred_metric = TT.from_lie_relative(predict, pred_pose0)
                    
                else:
                    trajectory_fact_metric = trajectory_fact_metric.extend_lie_relative(y_val)
                    trajectory_pred_metric = trajectory_pred_metric.extend_lie_relative(predict)

            if (step + 1) % self.window_size == 0:
                hidden = None
                pred_pose0 = None
                self.lm_count_trajectory += 1
                self.path_lengh = trajectory_fact_metric.path_length()

                loss_t_metric_global = self.loss_func_trajectory(trajectory_pred_metric, trajectory_fact_metric)
                p_mean_loc = translation_rmse_drift(self.loss_func_pose.pos_loss, self.path_lengh)
                r_mean_loc = rotation_rmse_drift(self.loss_func_pose.r_loss, self.path_lengh)

                self.loss_mean_test_trajectory = 1 / self.lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / self.lm_count_trajectory) * self.loss_mean_test_trajectory
                self.p_mean = 1 / self.lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / self.lm_count_trajectory) * self.p_mean
                self.r_mean = 1 / self.lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / self.lm_count_trajectory) * self.r_mean
                
            self.lm_count += 1
            self.loss_pose_mean_test = 1 / self.lm_count * loss_pose.item() + (1 - 1 / self.lm_count) * self.loss_pose_mean_test
            
            loss = self.weight_pose  * loss_pose + self.weight_trajectory * loss_t_metric_global
            self.loss_mean_test = 1 / self.lm_count * loss.item() + (1 - 1 / self.lm_count) * self.loss_mean_test
            
            val_bar.set_postfix({
            'loss': self.loss_mean_test,
            'loss_pose': self.loss_pose_mean_test,
            'loss_trajectory_gl': self.loss_mean_test_trajectory,
            'AT RMSE drift': self.p_mean,
            'AR RMSE drift': self.r_mean
            })
    
    
    def __call__(self):
        
        for epoch in range(self._start, self._end):
    
            if epoch > 5 and epoch % 3 == 0:
                self.self.weight_trajectory += 0.05
            
            self._init_metrices()

            self._train(epoch)
            self._test(epoch)
            
            self.dct_of_results['epoch'].append(epoch)
            self.dct_of_results['loss_pose_train'].append(self.loss_mean_train)
            self.dct_of_results['loss_pose_test'].append(self.loss_mean_test)
            self.dct_of_results['loss_trajectory_train'].append(self.loss_mean_train_trajectory)
            self.dct_of_results['loss_trajectory_train_gl'].append(self.loss_mean_trajectory_metric)
            self.dct_of_results['loss_trajectory_test_gl'].append(self.loss_mean_test_trajectory)
            self.dct_of_results['Average Translational RMSE drift'].append(self.p_mean)
            self.dct_of_results['Average Rotational RMSE drift'].append(self.r_mean)
            
            with open(self.path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
                json.dump(self.dct_of_results, w, ensure_ascii=False, indent=4)
            
            # TODO Действительно ли это критерий отбора?
            if self.p_mean + self.r_mean <= best_score:
                best_score = self.p_mean + self.r_mean
                torch.save(self.model.state_dict(), self.name_of_model)
                print('Модель сохранена')
                
        return self.dct_of_results
            
        
        
        

    
    