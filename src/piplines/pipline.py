import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm

from src.metrics.KITTI_metrics import translation_rmse_drift, rotation_rmse_drift
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

from src.function_of_loss.mse_pose import PoseLossTrajectory, PoseLoss



class TrainObject:
    
    def __init__(self, hidden, pred_pose0, localLoss, globalLoss, optimizer, weight_pose, weight_trajectory, window_size, device):
        
        self.hidden = hidden
        self.pred_pose0 = pred_pose0
        
        self.loss = localLoss
        self.loss_func_trajectory = globalLoss
        
        self.optimizer = optimizer
        
        self.weight_pose = weight_pose
        self.weight_trajectory = weight_trajectory
        self.window_size = window_size
        
        self.device = device 
    
    
    def __call__(self, out, model):
        '''
        Инициализация параметров обучения
        '''
        
        self.x_train = out[0].to(self.device) 
        self.imu = out[1].to(self.device) 
        self.y_train = out[2].to(self.device)
        pose = out[3].to(self.device)
        self.pose = PT.from_lie(pose)
        
        self.model = model
        
    def get_predict(self, dct):
        '''
        Получение предикта
        '''
        
        if self.hidden is not None:
            self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict, hidden = self.model(dct['x_train'].to(dtype=torch.float32), dct['imu'].to(dtype=torch.float32), self.hidden) # 6D Вектор алгебры Ли 

        self.predict = predict
        self.hidden = hidden
        
    def take_predict_after_norm(self, predict):
        self.predict_denorm = predict
    
    def get_localLoss(self, lm_count):
        '''
        Рассчет локальной ошибки
        '''
        
        predict = self.predict
        
        # Блок рассчета ошибки позы
        loss_pose = self.loss(predict, self.y_train)
        
        self.loss_pose_mean = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * self.loss_pose_mean
        self.loss_pose = loss_pose
        
    def get_globalLossMetrice(self):
        '''
        Рассчет метрики ошибки глобальной траектории на последовательности
        '''
        predict = self.predict_denorm
        pred_pose0 = self.pred_pose0
        
        if pred_pose0 is None:
            pred_pose0 = self.pose
            self.trajectory_fact_metric = TT.from_lie_relative(self.y_train, pred_pose0)
            self.trajectory_pred_metric = TT.from_lie_relative(predict.detach(), pred_pose0)
        else:
            trajectory_fact_metric = self.trajectory_fact_metric
            trajectory_pred_metric = self.trajectory_pred_metric
            
            pred_pose0 = pred_pose0.detach() # Отрубаем предыдущую позу от графа вычислений
            self.trajectory_fact_metric = trajectory_fact_metric.extend_lie_relative(self.y_train)
            self.trajectory_pred_metric = trajectory_pred_metric.extend_lie_relative(predict.detach())

        with torch.no_grad():
            loss_t_metric_global = self.loss_func_trajectory(trajectory_pred_metric[:-1], trajectory_fact_metric[:-1])
        
        self.loss_mean_trajectory_metric = 1 / self.lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / self.lm_count_trajectory) * self.loss_mean_trajectory_metric
        
        self.trajectory_fact_metric = trajectory_fact_metric
        self.trajectory_pred_metric = trajectory_pred_metric
        self.pred_pose0 = pred_pose0
        
        
    def get_globalLoss(self, lm_count_trajectory):
        
        predict = self.predict_denorm
        
        fact_pose = self.pose
        pred_pose0 = self.pred_pose0
        
        trajectory_fact = TT.from_absolute(fact_pose)
        trajectory = TT.from_lie_relative(predict, pred_pose0)    
        pred_pose0 = trajectory[-2].poses # Сохранили старую позу
        
        loss_t = self.loss_func_trajectory(trajectory[:-1], trajectory_fact)
        self.loss_mean_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory

        self.pred_pose0 = pred_pose0
        self.loss_t = loss_t
        
    def step_grad(self, step, lm_count):
        loss_pose = self.loss_pose
        loss_translation = self.loss_t
        
        loss = self.weight_pose  * loss_pose
            
        if (step + 1) % self.window_size == 0:
            self.hidden = None
            self.pred_pose0 = None
            loss += self.weight_trajectory * loss_translation
        
        loss.backward() # TODO нужно ли масштабирование?
        self.loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * self.loss_mean
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        


























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
    
    def _metrices_test(self, step, dct):
        trajectory_fact_metric = dct['trajectory_fact_metric']
        trajectory_pred_metric = dct['trajectory_pred_metric']
        
        
        if (step + 1) % self.window_size == 0:
            dct['hidden'] = None
            dct['pred_pose0'] = None
            
            self.path_lengh = trajectory_fact_metric.path_length()

            loss_t_metric_global = self.loss_func_trajectory(trajectory_pred_metric, trajectory_fact_metric)
            p_mean_loc = translation_rmse_drift(self.loss_func_pose.pos_loss, self.path_lengh)
            r_mean_loc = rotation_rmse_drift(self.loss_func_pose.r_loss, self.path_lengh)

            self.loss_mean_test_trajectory = 1 / self.lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / self.lm_count_trajectory) * self.loss_mean_test_trajectory
            self.p_mean = 1 / self.lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / self.lm_count_trajectory) * self.p_mean
            self.r_mean = 1 / self.lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / self.lm_count_trajectory) * self.r_mean
    
    
    def _train(self, epoch, dct):
        train_bar = tqdm(self.train_data, desc=f'Эпоха тренировочная {epoch+1}/{self._end}', position=0)
            
        self.model.train()
        
        train = TrainObject(
            hidden=None, 
            pred_pose0=None,
            localLoss=self.loss_func_pose,
            globalLoss=self.loss_func_trajectory,
            optimizer=self.optimizer,
            weight_pose=self.weight_pose,
            weight_trajectory=self.weight_trajectory,
            window_size=self.window_size,
            device=self.device
        )
        
        for step, out in enumerate(train_bar):
            
            
            train(out) # инициализировали параметры обучения
            train.get_predict() # Сделал прогноз
            
            self.lm_count += 1
            train.get_localLoss(self.lm_count)
            
            if self.normalize:
                predict = train.predict
                predict = self.normalize.denormalize(predict)
                train.take_predict_after_norm(predict)
            
            self.lm_count_trajectory += 1
            train.get_globalLoss
            dct = self._globalLoss(dct) # Ф-ия ошибки на окне
            
            dct = self._step_grad(step, dct)
            
            loss_mean_train = dct['loss_mean']
            loss_pose_mean_train = dct['loss_pose_mean']
            loss_mean_train_trajectory = dct['loss_mean_trajectory']
            loss_mean_trajectory_metric = dct['loss_mean_trajectory_metric']
            
            train_bar.set_postfix({
                'loss': loss_mean_train,
                'loss_pose': loss_pose_mean_train, 
                'loss_trajectory_win': loss_mean_train_trajectory,
                'loss_trajectory_gl': loss_mean_trajectory_metric
            })
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _test(self, epoch, dct):
        
        self.model.eval()
            
        val_bar = tqdm(self.test_data, desc=f'Эпоха валидационная {epoch+1}/{self._end}', position=1)
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
        dct['hidden'] = None
        dct['pred_pose0'] = None 
        dct['loss_t_metric_global'] = 0 # При инициализации
        
        # TODO доделать оконный метод
        for step, out in enumerate(val_bar):
            
            dct = self._init_params(out) # инициализировали параметры валидации
            
            with torch.no_grad():
                dct = self._predict(dct) # Сделали прогноз
                
                self.lm_count += 1
                dct = self._localLoss(dct) # Оптимизировали локальную ф-ию
                
                if self.normalize:
                    predict = dct['predict']
                    predict = self.normalize.denormalize(predict)
                    dct['predict'] = predict

                self.lm_count_trajectory += 1
                dct = self._globalLossMetrice(dct)
                
                
            
                
                
            
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
            
        
        
        

    
    