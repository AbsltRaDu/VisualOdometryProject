import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm

from src.metrics.KITTI_metrics import translation_rmse_drift, rotation_rmse_drift
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

from src.function_of_loss.mse_pose import PoseLossTrajectory, PoseLoss
from src.piplines.trainObject import CNNTrainerStep, RNNTrainerStep, RNNIMUTrainerStep

class TrainerCNN:
    
    def __init__(self, train_data, test_data, model, loss_func_pose, 
                loss_func_trajectory, optimizer, epochs, device, 
                name_of_model, path_to_save_process_of_fitting,
                normalize=None, best_score = 10**10, weight_pose=1, weight_trajectory=0, window_size=20):
        
        '''
        Класс гибкий пайплайн обучения сетей
        
        Вход:
            train_data: Тренировочный Dataloader
            test_data: Тестовый Dataloader
            model: модель сети
            loss_func_pose: Локальная ф-ия ошибок
            loss_func_trajectory: Глобальная ф-ия ошибок
            optimizer: Оптимизатор
            epochs: Кол-во эпох
            device: Девай CPU/CUDA
            name_of_model: Имя модели для сохранения
            path_to_save_process_of_fitting: Ссылка для сохранения результата обучения
            normalize: Объект класса нормализации/денормализации
            best_score: лучший результат дефолтный
            weight_trajectory: Вес глобальной ошибки в общей ф-ии
            weight_pose: Вес локальной ошибки в общей ф-ии
            window_size: Окно в рамках которого оценивается ф-ия ошиби траектории и сама траектория
            rnn_mode: Режим обучения rnn моделей
        '''
        
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
            
            p_hist = self.dct_of_results['Average Translational RMSE drift']
            r_hist = self.dct_of_results['Average Rotational RMSE drift']
            self.best_score = min(p + r for p, r in zip(p_hist, r_hist))
            
        else:   
            self.dct_of_results = defaultdict(list)
            self._start = 0
            self._end = epochs

    def init_trainObject(self):
        '''
        Инициализирует объект класса TrainObject
        
        По сути просто обертка, чтобы можно было менять в классе только
        класс TrainObject в зависимости от задачи
        '''
        train = CNNTrainerStep(
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

        return train    
    
    def _train(self, epoch):
        train_bar = tqdm(self.train_data, desc=f'Эпоха тренировочная {epoch+1}/{self._end}', position=0)
        len_train_bar = len(train_bar)
        
        self.model.train()
        
        train = self.init_trainObject()
        
        
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

            train.step_grad(step, self.lm_count, len_train_bar) # Делаю шаг градиента
            
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
        
        return train
    
    def _test(self, epoch):
        
        self.model.eval()
            
        val_bar = tqdm(self.test_data, desc=f'Эпоха валидационная {epoch+1}/{self._end}', position=1)
        len_val_bar = len(val_bar)
        
        self.lm_count = 0
        self.lm_count_trajectory = 0
        
        test = self.init_trainObject()
        
        # TODO доделать оконный метод
        for step, out in enumerate(val_bar):
            
            test(out) # Инициализировал параметры валидации
            
            with torch.no_grad():
                test.get_predict()
                
                self.lm_count += 1
                test.get_localLoss(self.lm_count)
                
                predict = test.predict
                if self.normalize:
                    predict = self.normalize.denormalize(predict)
                test.take_predict_after_norm(predict)

                self.lm_count_trajectory += 1
                test.get_globalLossMetrice(self.lm_count_trajectory)
                test.get_globalLoss(self.lm_count_trajectory)
                test.get_metrices_test(step, self.lm_count, self.lm_count_trajectory, len_val_bar)
                
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
    
    def _record_results_of_epoch(self, epoch, train, test):
        
        self.dct_of_results['epoch'].append(epoch)
        self.dct_of_results['loss_train'].append(train.loss_mean)
        self.dct_of_results['loss_test'].append(test.loss_mean)
        self.dct_of_results['loss_pose_train'].append(train.loss_pose_mean)
        self.dct_of_results['loss_pose_test'].append(test.loss_pose_mean)
        self.dct_of_results['loss_trajectory_train'].append(train.loss_mean_trajectory)
        self.dct_of_results['loss_trajectory_test'].append(test.loss_mean_trajectory)
        self.dct_of_results['loss_trajectory_train_gl'].append(train.loss_mean_trajectory_metric)
        self.dct_of_results['loss_trajectory_test_gl'].append(test.loss_mean_trajectory_metric)
        self.dct_of_results['Average Translational RMSE drift'].append(test.p_mean)
        self.dct_of_results['Average Rotational RMSE drift'].append(test.r_mean)
        
        with open(self.path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(self.dct_of_results, w, ensure_ascii=False, indent=4)
    
    def __call__(self):
        
        for epoch in range(self._start, self._end):
    
            # if epoch > 20 and epoch % 5 == 0:
            #     self.weight_trajectory += 0.05

            train = self._train(epoch) # Прогнали тренировочную выборку, обучение
            test = self._test(epoch) # Прогнали валидационную выборку
            
            self._record_results_of_epoch(epoch, train, test) # Записываем результат
            
            # TODO Действительно ли это критерий отбора?
            if test.p_mean + test.r_mean <= self.best_score:
                self.best_score = test.p_mean + test.r_mean
                torch.save(self.model.state_dict(), self.name_of_model)
                print('Модель сохранена')
                
        return self.dct_of_results
            
        
class TrainerRNN(TrainerCNN):
    
    def init_trainObject(self):
        '''
        Инициализирует объект класса TrainObject
        
        По сути просто обертка, чтобы можно было менять в классе только
        класс TrainObject в зависимости от задачи
        '''
        train = RNNTrainerStep(
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

        return train 

class TrainerRNNIMU(TrainerRNN):
    
    def init_trainObject(self):
        '''
        Инициализирует объект класса TrainObject
        
        По сути просто обертка, чтобы можно было менять в классе только
        класс TrainObject в зависимости от задачи
        '''
        train = RNNIMUTrainerStep(
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

        return train 
    
    