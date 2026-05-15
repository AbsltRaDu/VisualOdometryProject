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
        
        self.loss_pose_mean = 0
        self.p_mean_loc = 0
        self.r_mean_loc = 0
    
    
    def __call__(self, out, model):
        '''
        Инициализация параметров обучения
        '''
        
        self.x_train = out[0].to(self.device) 
        # self.imu = out[1].to(self.device) 
        self.y_train = out[2].to(self.device)
        pose = out[3].to(self.device)
        self.pose = PT.from_lie(pose)
        
        
    def get_predict(self, dct):
        '''
        Получение предикта
        '''
        
        # if self.hidden is not None:
        #     self.hidden = self.model.detach_hidden(self.hidden) # отрубили от графа вычислений предыдущий скрытый слой
        
        predict = self.model(self.x_train.to(dtype=torch.float32)) # 6D Вектор алгебры Ли 

        self.predict = predict
        # self.hidden = hidden
        
    def take_predict_after_norm(self, predict):
        self.predict_denorm = predict
    
    def get_localLoss(self, lm_count):
        '''
        Рассчет локальной ошибки
        '''
        
        predict = self.predict
        
        # Блок рассчета ошибки позы
        loss_pose = self.loss_func(predict, self.y_train)
        
        self.loss_pose_mean = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * self.loss_pose_mean
        self.loss_pose = loss_pose
        
        self.p_mean_loc = 1 / lm_count * self.loss_pose.pos_loss.item() + (1 - 1 / lm_count) * self.p_mean_loc
        self.r_mean_loc = 1 / lm_count * self.loss_pose.r_loss.item() + (1 - 1 / lm_count) * self.r_mean_loc 
        
        
    def get_globalLossMetrice(self, lm_count_trajectory):
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
            loss_t_metric_global = self.loss_func_trajectory(self.trajectory_pred_metric[:-1], self.trajectory_fact_metric[:-1])
        
        # TODO Нужно ли тут усреднее?
        self.loss_mean_trajectory_metric = 1 / lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / lm_count_trajectory) * self.loss_mean_trajectory_metric
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
        
    def get_metrices_test(self, step, lm_count_trajectory):
        '''
        Рассчет основных метрик на валидации
        '''
        
        trajectory_fact_metric = self.trajectory_fact_metric
        trajectory_pred_metric = self.trajectory_pred_metric
        
        
        if (step + 1) % self.window_size == 0:
            self.hidden = None
            self.pred_pose0 = None
            
            self.path_lengh = trajectory_fact_metric.path_length()

            p_mean_loc = translation_rmse_drift(self.p_mean_loc , self.path_lengh)
            r_mean_loc = rotation_rmse_drift(self.r_mean_loc, self.path_lengh)

            self.p_mean_loc = 0
            self.r_mean_loc = 0
            
            self.p_mean = 1 / lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.p_mean
            self.r_mean = 1 / lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * self.r_mean







class trainingProgressive:
    
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
        

    
    def _train(self, epoch, dct):
        train_bar = tqdm(self.train_data, desc=f'Эпоха тренировочная {epoch+1}/{self._end}', position=0)
            
        self.model.train()
        
        train = TrainObject(
            model=self.model,
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
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
        for step, out in enumerate(train_bar):
            
            train(out) # инициализировали параметры обучения
            train.get_predict() # Сделал прогноз
            
            self.lm_count += 1
            train.get_localLoss(self.lm_count)
            
            predict = train.predict
            if self.normalize:
                predict = self.normalize.denormalize(predict)
            train.take_predict_after_norm(predict)
            
            self.lm_count_trajectory += 1
            train.get_globalLossMetrice(self.lm_count_trajectory) # Считаю метрику глобальной траектории
            train.get_globalLoss(self.lm_count_trajectory) # Считаю ошибку на окне

            train.step_grad(step, self.lm_count) # Делаю шаг градиента
            
            loss_mean = train.loss_mean
            loss_pose = train.loss_pose_mean
            loss_mean_trajectory = train.loss_mean_trajectory
            loss_mean_trajectory_metric = train.loss_mean_trajectory_metric
            
            train_bar.set_postfix({
                'loss': loss_mean,
                'loss_pose': loss_pose, 
                'loss_trajectory_win': loss_mean_trajectory,
                'loss_trajectory_gl': loss_mean_trajectory_metric
            })
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return train
    
    def _test(self, epoch, dct):
        
        self.model.eval()
            
        val_bar = tqdm(self.test_data, desc=f'Эпоха валидационная {epoch+1}/{self._end}', position=1)
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
        test = TrainObject(
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
        
        # TODO доделать оконный метод
        for step, out in enumerate(val_bar):
            
            test(out) # Инициализировал параметры валидации
            
            with torch.no_grad():
                test.get_predict()
                
                self.lm_count += 1
                test.get_localLoss(self.lm_count)
                
                if self.normalize:
                    predict = test.predict
                    predict = self.normalize.denormalize(predict)
                    test.take_predict_after_norm(predict)

                self.lm_count_trajectory += 1
                test.get_globalLossMetrice(self.lm_count_trajectory)
                test.get_globalLoss(self.lm_count_trajectory)
                test.get_metrices_test(step, self.lm_count_trajectory)
                
            loss_mean = test.loss_mean
            loss_pose_mean = test.loss_pose_mean
            lose_trajectory_mean = test.loss_mean_trajectory
            loss_trajectory_global = test.loss_mean_trajectory_metric
            p_mean = test.p_mean
            r_mean = test.r_mean
            
            
            val_bar.set_postfix({
            'loss': loss_mean,
            'loss_pose': loss_pose_mean,
            'loss_trajectory': lose_trajectory_mean,
            'loss_trajectory_gl': loss_trajectory_global,
            'AT RMSE drift': p_mean,
            'AR RMSE drift': r_mean
            })

        return test
    
    def __call__(self):
        
        for epoch in range(self._start, self._end):
    
            if epoch > 20 and epoch % 5 == 0:
                self.weight_trajectory += 0.05
            
            self._init_metrices()

            train = self._train(epoch)
            test = self._test(epoch)
            
            self.dct_of_results['epoch'].append(epoch)
            self.dct_of_results['loss_train'].append(train.loss_mean)
            self.dct_of_results['loss_test'].append(test.loss_mean)
            self.dct_of_results['loss_pose_train'].append(train.loss_pose)
            self.dct_of_results['loss_pose_test'].append(test.loss_pose)
            self.dct_of_results['loss_trajectory_train'].append(train.loss_mean_trajectory)
            self.dct_of_results['loss_trajectory_train'].append(test.loss_mean_trajectory)
            self.dct_of_results['loss_trajectory_train_gl'].append(train.loss_mean_trajectory_metric)
            self.dct_of_results['loss_trajectory_test_gl'].append(test.loss_mean_trajectory_metric)
            self.dct_of_results['Average Translational RMSE drift'].append(test.p_mean)
            self.dct_of_results['Average Rotational RMSE drift'].append(test.r_mean)
            
            with open(self.path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
                json.dump(self.dct_of_results, w, ensure_ascii=False, indent=4)
            
            # TODO Действительно ли это критерий отбора?
            if test.p_mean + test.r_mean <= self.best_score:
                best_score = self.p_mean + self.r_mean
                torch.save(self.model.state_dict(), self.name_of_model)
                print('Модель сохранена')
                
        return self.dct_of_results
            
        
        
        

    
    