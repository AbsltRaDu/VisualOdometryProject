from collections import defaultdict
import os
import json

import torch
import torch.nn as nn 

from tqdm import tqdm

from src.metrics.KITTI_metrics import translation_rmse_drift, rotation_rmse_drift
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.TrajectoryTorch import TrajectoryTorch as TT

from src.function_of_loss.mse_pose import PoseLossTrajectory, PoseLoss

def training_CNN(train_data, test_data, model, loss_func_pose: PoseLoss, loss_func_trajectory: PoseLossTrajectory, optimizer, epochs, device, name_of_model, path_to_save_process_of_fitting, squueze=False, normalize=None):
    best_score = 10**10
    
    # Блок подготовки словаря для записи
    if os.path.isfile(path_to_save_process_of_fitting):
        with open(path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
            dct_of_results = json.load(f)
            
        _start = dct_of_results['epoch'][-1] + 1
        _end = _start + epochs
        
        best_score = dct_of_results['Average Translational RMSE drift'][-1] + dct_of_results['Average Rotational RMSE drift'][-1]
        
    else:   
        dct_of_results = defaultdict(list)
        _start = 0
        _end = epochs
    
    for epoch in range(_start, _end):
    
        loss_mean_train = 0
        loss_mean_train_trajectory = 0
        
        loss_mean_test = 0
        loss_mean_test_trajectory = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        lm_count = 0

        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{_end}', position=0)
        
        model.train()
        
        for x_train, y_train, pose in train_bar:
            x_train = x_train.to(device) 
            y_train = y_train.to(device)
            pose = pose.to(device)
            
            predict = model(x_train) # 6D Вектор алгебры Ли 
            predict = predict.unsqueeze(0) if squueze else predict
            
            loss = loss_func_pose(predict, y_train)
            lm_count += 1
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            if normalize:
                predict = normalize.denormalize(predict)
            
            pose_fact = PT.from_lie(pose)
            trajectory = TT.from_lie_relative(predict, pose_fact[0])
            
            loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
            loss_mean_train_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_train_trajectory
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix({
                'loss_pose': loss_mean_train,
                'loss_trajectory': loss_mean_train_trajectory
            })
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{_end}', position=1)
        
        lm_count = 0
        
        for x_val, y_val, pose in val_bar:
            x_val = x_val.to(device) 
            y_val = y_val.to(device)
            pose = pose.to(device)
            
            with torch.no_grad():
                predict = model(x_val)
                predict = predict.unsqueeze(0) if squueze else predict
                
                loss = loss_func_pose(predict, y_val)
                
                if normalize:
                    predict = normalize.denormalize(predict)
            
                pose_fact = PT.from_lie(pose)
                trajectory = TT.from_lie_relative(predict, pose_fact[0])
                
                loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
                
                path_lengh = trajectory.path_length()
                
                p_mean_loc = translation_rmse_drift(loss_func_pose.pos_loss, path_lengh)
                r_mean_loc = rotation_rmse_drift(loss_func_pose.r_loss, path_lengh)
                
                lm_count += 1
                loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
                loss_mean_test_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_test_trajectory
                p_mean = 1 / lm_count * p_mean_loc.mean().item() + (1 - 1 / lm_count) * p_mean
                r_mean = 1 / lm_count * r_mean_loc.mean().item() + (1 - 1 / lm_count) * r_mean

                val_bar.set_postfix({
                'loss_pose': loss_mean_test,
                'loss_trajectory': loss_mean_test_trajectory,
                'AT RMSE drift': p_mean,
                'AR RMSE drift': r_mean
                })

        dct_of_results['epoch'].append(epoch)
        dct_of_results['loss_pose_train'].append(loss_mean_train)
        dct_of_results['loss_pose_test'].append(loss_mean_test)
        dct_of_results['loss_trajectory_train'].append(loss_mean_train_trajectory)
        dct_of_results['loss_trajectory_test'].append(loss_mean_test_trajectory)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        with open(path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
        
        # TODO Действительно ли это критерий отбора?
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results






















def training_CNN_progressive(train_data, test_data, model, loss_func_pose: PoseLoss, 
                             loss_func_trajectory: PoseLossTrajectory, optimizer, epochs, device, 
                             name_of_model, path_to_save_process_of_fitting,
                             squueze=False, normalize=None, window_size: int = 10):
    best_score = 10**10
    
    # Блок подготовки словаря для записи
    if os.path.isfile(path_to_save_process_of_fitting):
        with open(path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
            dct_of_results = json.load(f)
            
        _start = dct_of_results['epoch'][-1] + 1
        _end = _start + epochs
        
        best_score = dct_of_results['Average Translational RMSE drift'][-1] + dct_of_results['Average Rotational RMSE drift'][-1]
        
    else:   
        dct_of_results = defaultdict(list)
        _start = 0
        _end = epochs
    
    for epoch in range(_start, _end):
    
        loss_mean_train = 0
        loss_mean_train_trajectory = 0
        
        loss_mean_test = 0
        loss_mean_test_trajectory = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        
        lm_count = 0
        lm_count_trajectory = 0

        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{_end}', position=0)
        
        model.train()
        predict_window = None
        
        for step, (img1, img2, y_train, pose) in enumerate(train_bar):
            img1 = img1.to(device) 
            img2 = img2.to(device) 
            y_train = y_train.to(device)
            pose = pose.to(device)
            
            predict = model(img1, img2) # 6D Вектор алгебры Ли 
            predict = predict.unsqueeze(0) if squueze else predict
            
            loss = loss_func_pose(predict, y_train)
            lm_count += 1
            
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            if normalize:
                predict = normalize.denormalize(predict)
            
            if predict_window is None:
                predict_window = predict.unsqueeze(1).detach() # добавляем ось
                pose_window = pose.unsqueeze(1)
            else:
                predict_window = torch.cat([predict_window, predict.unsqueeze(1).detach()], dim=1) # Не учитываем накопленное окно в графе вычислений, мы не оптимизируем тут траекторию
                pose_window = torch.cat([pose_window, pose.unsqueeze(1)], dim=1)
            
            if (step + 1) % window_size == 0:
                lm_count_trajectory += 1
                
                pose_fact = PT.from_lie(pose_window)
                trajectory = TT.from_lie_relative(predict_window, pose_fact[0])
                
                loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
                loss_mean_train_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * loss_mean_train_trajectory
                
                predict_window = None 
                pose_window = None
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix({
                'loss_pose': loss_mean_train,
                'loss_trajectory': loss_mean_train_trajectory
            })
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{_end}', position=1)
        
        lm_count = 0
        lm_count_trajectory = 0
        
        predict_window = None 
        pose_window = None
        
        for step, (img1, img2, y_val, pose) in enumerate(val_bar):
            img1 = img1.to(device) 
            img2 = img2.to(device) 
            y_val = y_val.to(device)
            pose = pose.to(device)
            
            with torch.no_grad():
                predict = model(img1, img2)
                predict = predict.unsqueeze(0) if squueze else predict
                
                loss = loss_func_pose(predict, y_val)
                
                if normalize:
                    predict = normalize.denormalize(predict)

                if predict_window is None:
                    predict_window = predict.unsqueeze(1) # добавляем ось
                    pose_window = pose.unsqueeze(1)
                else:
                    predict_window = torch.cat([predict_window, predict.unsqueeze(1)], dim=1)
                    pose_window = torch.cat([pose_window, pose.unsqueeze(1)], dim=1)
                
                if (step + 1) % window_size == 0:
                    lm_count_trajectory += 1
                    
                    pose_fact = PT.from_lie(pose_window)
                    trajectory = TT.from_lie_relative(predict_window, pose_fact[0])
                    
                    loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
                    loss_mean_test_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * loss_mean_test_trajectory

                    path_lengh = trajectory.path_length()
                
                    p_mean_loc = translation_rmse_drift(loss_func_pose.pos_loss, path_lengh)
                    r_mean_loc = rotation_rmse_drift(loss_func_pose.r_loss, path_lengh)

                    p_mean = 1 / lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * p_mean
                    r_mean = 1 / lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * r_mean

                    predict_window = None 
                    pose_window = None
                    
                lm_count += 1
                loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
                
                val_bar.set_postfix({
                'loss_pose': loss_mean_test,
                'loss_trajectory': loss_mean_test_trajectory,
                'AT RMSE drift': p_mean,
                'AR RMSE drift': r_mean
                })

        dct_of_results['epoch'].append(epoch)
        dct_of_results['loss_pose_train'].append(loss_mean_train)
        dct_of_results['loss_pose_test'].append(loss_mean_test)
        dct_of_results['loss_trajectory_train'].append(loss_mean_train_trajectory)
        dct_of_results['loss_trajectory_test'].append(loss_mean_test_trajectory)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        with open(path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
        
        # TODO Действительно ли это критерий отбора?
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results 

def training_RCNN_progressive_JointTrain(train_data, test_data, model, loss_func_pose: PoseLoss, 
                             loss_func_trajectory: PoseLossTrajectory, optimizer, epochs, device, 
                             name_of_model, path_to_save_process_of_fitting,
                             squueze=False, normalize=None, window_size: int = 10, 
                             weight_pose: int = 1, weight_trajectory: int = 0.05):
    best_score = 10**10
    
    # Блок подготовки словаря для записи
    if os.path.isfile(path_to_save_process_of_fitting):
        with open(path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
            dct_of_results = json.load(f)
            
        _start = dct_of_results['epoch'][-1] + 1
        _end = _start + epochs
        
        best_score = dct_of_results['Average Translational RMSE drift'][-1] + dct_of_results['Average Rotational RMSE drift'][-1]
        
    else:   
        dct_of_results = defaultdict(list)
        _start = 0
        _end = epochs
    
    for epoch in range(_start, _end):
        
        if epoch < 10:
            weight_trajectory = weight_trajectory * (epoch + 1)
        
        loss_mean_train = 0
        loss_pose_mean_train = 0
        loss_mean_train_trajectory = 0
        loss_mean_trajectory_metric = 0
        
        loss_mean_test = 0
        loss_pose_mean_test = 0
        loss_mean_test_trajectory = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        
        lm_count = 0
        lm_count_trajectory = 0

        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{_end}', position=0)
        
        model.train()
        
        hidden = None
        pred_pose0 = None
        
        for step, (x_train, y_train, pose) in enumerate(train_bar):
            x_train = x_train.to(device) 
            y_train = y_train.to(device)
            pose = pose.to(device)
            
            if hidden is not None:
                hidden = model.detach_hidden(hidden) # отрубили от графа вычислений предыдущий скрытый слой
            
            predict, hidden = model(x_train, hidden) # 6D Вектор алгебры Ли 
            predict = predict.unsqueeze(0) if squueze else predict
            
            loss_pose = loss_func_pose(predict, y_train)
            lm_count += 1
            loss_pose_mean_train = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_pose_mean_train
            
            if normalize:
                predict = normalize.denormalize(predict)
            
            lm_count_trajectory += 1
                
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
            
            loss_t = loss_func_trajectory(trajectory[:-1], trajectory_fact)
            loss_mean_train_trajectory = 1 / lm_count_trajectory * loss_t.item() + (1 - 1 / lm_count_trajectory) * loss_mean_train_trajectory

            # Чтоб не дай бог градиент не посчитался по всей последовательности
            with torch.no_grad():
                loss_t_metric_global = loss_func_trajectory(trajectory_pred_metric, trajectory_fact_metric)
            loss_mean_trajectory_metric = 1 / lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / lm_count_trajectory) * loss_mean_trajectory_metric
            
            loss = weight_pose * loss_pose + weight_trajectory * loss_t
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            (loss / window_size).backward() # Делим на длину окна, что масштабировать градиенты
            
            if (step + 1) % window_size == 0:
                hidden = None
                pred_pose0 = None
                
                optimizer.step()
                optimizer.zero_grad()

            
            train_bar.set_postfix({
                'loss': loss_mean_train,
                'loss_pose': loss_pose_mean_train,
                'loss_trajectory_win': loss_mean_train_trajectory,
                'loss_trajectory_gl': loss_mean_trajectory_metric
            })
        
        optimizer.step()
        optimizer.zero_grad() 
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{_end}', position=1)
        
        lm_count = 0
        lm_count_trajectory = 0
        
        hidden = None
        pred_pose0 = None 
        loss_t_metric_global = 0 # При инициализации
        
        # TODO доделать оконный метод
        for step, (x_val, y_val, pose) in enumerate(val_bar):
            x_val = x_val.to(device) 
            y_val = y_val.to(device)
            pose = pose.to(device)
            
            with torch.no_grad():
                predict, hidden = model(x_val, hidden)
                predict = predict.unsqueeze(0) if squueze else predict
                
                loss_pose = loss_func_pose(predict, y_val)
                
                if normalize:
                    predict = normalize.denormalize(predict)

                
                fact_pose = PT.from_lie(pose)
                if pred_pose0 is None:
                    pred_pose0 = fact_pose[0]
                    trajectory_fact_metric = TT.from_lie_relative(y_train, pred_pose0)
                    trajectory_pred_metric = TT.from_lie_relative(predict, pred_pose0)
                    
                else:
                    trajectory_fact_metric = trajectory_fact_metric.extend_lie_relative(y_train)
                    trajectory_pred_metric = trajectory_pred_metric.extend_lie_relative(predict)
            
            
            
            lm_count_trajectory += 1
            if (step + 1) % window_size == 0:
                hidden = None
                path_lengh = trajectory.path_length()

                loss_t_metric_global = loss_func_trajectory(trajectory_pred_metric, trajectory_fact_metric)
                p_mean_loc = translation_rmse_drift(loss_func_pose.pos_loss, path_lengh)
                r_mean_loc = rotation_rmse_drift(loss_func_pose.r_loss, path_lengh)

                loss_mean_test_trajectory = 1 / lm_count_trajectory * loss_t_metric_global.item() + (1 - 1 / lm_count_trajectory) * loss_mean_test_trajectory
                p_mean = 1 / lm_count_trajectory * p_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * p_mean
                r_mean = 1 / lm_count_trajectory * r_mean_loc.mean().item() + (1 - 1 / lm_count_trajectory) * r_mean
                
            lm_count += 1
            loss_pose_mean_test = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_pose_mean_test
            
            loss = weight_pose * loss_pose + weight_trajectory * loss_t_metric_global
            loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
            
            val_bar.set_postfix({
            'loss': loss_mean_test,
            'loss_pose': loss_pose_mean_test,
            'loss_trajectory_gl': loss_mean_test_trajectory,
            'AT RMSE drift': p_mean,
            'AR RMSE drift': r_mean
            })

        dct_of_results['epoch'].append(epoch)
        dct_of_results['loss_pose_train'].append(loss_mean_train)
        dct_of_results['loss_pose_test'].append(loss_mean_test)
        dct_of_results['loss_trajectory_train'].append(loss_mean_train_trajectory)
        dct_of_results['loss_trajectory_train_gl'].append(loss_mean_trajectory_metric)
        dct_of_results['loss_trajectory_test_gl'].append(loss_mean_test_trajectory)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        with open(path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
        
        # TODO Действительно ли это критерий отбора?
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results




















def training_CNN_JointTraning(train_data, test_data, model, loss_func_pose: PoseLoss, loss_func_trajectory: PoseLossTrajectory, optimizer, epochs, device, name_of_model, path_to_save_process_of_fitting,
                              squueze=False, normalize=None, weigth_local=1, weigth_trajectory=0):
    best_score = 10**10
    
    # Блок подготовки словаря для записи
    if os.path.isfile(path_to_save_process_of_fitting):
        with open(path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
            dct_of_results = json.load(f)
            
        _start = dct_of_results['epoch'][-1] + 1
        _end = _start + epochs
            
    else:   
        dct_of_results = defaultdict(list)
        _start = 0
        _end = epochs
    
    for epoch in range(_start, _end):
        
        # if epoch % 20 == 0 and epoch != 0:
            # weigth_local -= 0.05
        weigth_trajectory += 0.01
        
        
        loss_mean_train = 0
        loss_mean_train_trajectory = 0
        loss_mean_train_pose = 0
        
        loss_mean_test = 0
        loss_mean_test_pose = 0
        loss_mean_test_trajectory = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        lm_count = 0

        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{_end}', position=0)
        
        model.train()
        
        for x_train, y_train, pose in train_bar:
            x_train = x_train.to(device) 
            y_train = y_train.to(device)
            pose = pose.to(device)
            
            predict = model(x_train) # 6D Вектор алгебры Ли 
            predict = predict.unsqueeze(0) if squueze else predict
            
            loss_pose = loss_func_pose(predict, y_train)
            lm_count += 1
            loss_mean_train_pose = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_mean_train_pose
            
            if normalize:
                predict = normalize.denormalize(predict)
            
            pose_fact = PT.from_lie(pose)
            trajectory = TT.from_lie_relative(predict, pose_fact[0])
            
            loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
            loss_mean_train_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_train_trajectory
            
            loss = weigth_local * loss_pose + weigth_trajectory * loss_t
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix({
                'loss': loss_mean_train,
                'loss_pose': loss_mean_train_pose,
                'loss_trajectory': loss_mean_train_trajectory
            })
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{_end}', position=1)
        
        lm_count = 0
        
        for x_val, y_val, pose in val_bar:
            x_val = x_val.to(device) 
            y_val = y_val.to(device)
            pose = pose.to(device)
            
            with torch.no_grad():
                predict = model(x_val)
                predict = predict.unsqueeze(0) if squueze else predict
                
                loss_pose = loss_func_pose(predict, y_val)
                
                if normalize:
                    predict = normalize.denormalize(predict)
            
                pose_fact = PT.from_lie(pose)
                trajectory = TT.from_lie_relative(predict, pose_fact[0])
                
                loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
                
                
                loss = weigth_local * loss_pose + weigth_trajectory * loss_t
                
                path_lengh = trajectory.path_length()
                
                p_mean_loc = translation_rmse_drift(loss_func_pose.pos_loss, path_lengh)
                r_mean_loc = rotation_rmse_drift(loss_func_pose.r_loss, path_lengh)
                
                lm_count += 1
                loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
                loss_mean_test_pose = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_mean_test_pose
                loss_mean_test_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_test_trajectory
                p_mean = 1 / lm_count * p_mean_loc.mean().item() + (1 - 1 / lm_count) * p_mean
                r_mean = 1 / lm_count * r_mean_loc.mean().item() + (1 - 1 / lm_count) * r_mean

                val_bar.set_postfix({
                'loss': loss_mean_test,
                'loss_pose': loss_mean_test_pose,
                'loss_trajectory': loss_mean_test_trajectory,
                'AT RMSE drift': p_mean,
                'AR RMSE drift': r_mean
                })

        dct_of_results['epoch'].append(epoch)
        dct_of_results['lose_train'].append(loss_mean_train),
        dct_of_results['lose_test'].append(loss_mean_test),
        dct_of_results['loss_pose_train'].append(loss_mean_train_pose)
        dct_of_results['loss_pose_test'].append(loss_mean_test_pose)
        dct_of_results['loss_trajectory_train'].append(loss_mean_train_trajectory)
        dct_of_results['loss_trajectory_test'].append(loss_mean_test_trajectory)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        with open(path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
        
        # TODO Действительно ли это критерий отбора?
        # Выяснилось, что при отсутствии нормализации - Да
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results

def training_RCNN_JointTraning(train_data, test_data, model, loss_func_pose: PoseLoss, loss_func_trajectory: PoseLossTrajectory, 
                               optimizer, epochs, device, name_of_model, path_to_save_process_of_fitting,
                              squueze=False, normalize=None, weigth_local=1, weigth_trajectory=0, accum_steps: int = 1000):
    best_score = 10**10
    
    # Блок подготовки словаря для записи
    if os.path.isfile(path_to_save_process_of_fitting):
        with open(path_to_save_process_of_fitting, 'r', encoding='utf-8') as f:
            dct_of_results = json.load(f)
            
        _start = dct_of_results['epoch'][-1] + 1
        _end = _start + epochs
            
    else:   
        dct_of_results = defaultdict(list)
        _start = 0
        _end = epochs
    
    for epoch in range(_start, _end):
        
        if epoch % 5 == 0 and epoch != 0:
            weigth_trajectory += 0.01
        
        prev_dataset_ids = None
        
        loss_mean_train = 0
        loss_mean_train_trajectory = 0
        loss_mean_train_pose = 0
        
        loss_mean_test = 0
        loss_mean_test_pose = 0
        loss_mean_test_trajectory = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        lm_count = 0

        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{_end}', position=0)
        
        model.train()
        optimizer.zero_grad() 

        # TODO нужно реализовать перемешивание датасетов для каждой эпохи, чтобы их последовательность не была одинаковой
        for step, (x_train, y_train, pose, dataset_ids) in enumerate(train_bar):
            x_train = x_train.to(device) 
            y_train = y_train.to(device)
            pose = pose.to(device)
            
            if prev_dataset_ids is None: # Для старта
                hidden = None
                pred_pose0 = None
            else:
                changed = dataset_ids != prev_dataset_ids # Если в батче хотя бы одна последовательность уже из другого датасета

                if changed.any(): # Сбрасываем hidden и стартовую позу
                    if step % accum_steps != 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    hidden = None
                    pred_pose0 = None

            prev_dataset_ids = dataset_ids.clone()
                    
            if hidden is not None:
                hidden = model.detach_hidden(hidden) # отсоединяем предыдущую итерацию от графа вычислений
            
            predict, hidden = model(x_train, hidden) # 6D Вектор алгебры Ли 
            predict = predict.unsqueeze(0) if squueze else predict
            
            loss_pose = loss_func_pose(predict, y_train)
            lm_count += 1
            loss_mean_train_pose = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_mean_train_pose
            
            if normalize:
                predict = normalize.denormalize(predict)
            
            pose_fact = PT.from_lie(pose)
            
            if pred_pose0 is None:
                pred_pose0 = pose_fact[0]  # стартуем от реальной первой позы
            else:
                pred_pose0 = pred_pose0.detach()
            
            trajectory = TT.from_lie_relative(predict, pred_pose0)

            loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
            loss_mean_train_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_train_trajectory
            
            pred_pose0 = trajectory.last_pose()
            
            loss = weigth_local * loss_pose + weigth_trajectory * loss_t
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            (loss / accum_steps).backward()
            
            if (step + 1) % accum_steps == 0: # Считаем градиенты, если только наши окна прошли полную последовательность
                optimizer.step()
                optimizer.zero_grad()
            
            train_bar.set_postfix({
                'loss': loss_mean_train,
                'loss_pose': loss_mean_train_pose,
                'loss_trajectory': loss_mean_train_trajectory
            })
            
        if (step + 1) % accum_steps != 0: # на случай, если последовательность не бьыла завершена
            optimizer.step()
            optimizer.zero_grad()
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{_end}', position=1)
        
        lm_count = 0
        
        for x_val, x_val, pose, dataset_ids in val_bar:
            x_val = x_val.to(device) 
            y_val = y_val.to(device)
            pose = pose.to(device)
            
            with torch.no_grad():
                predict, _ = model(x_val)
                predict = predict.unsqueeze(0) if squueze else predict
                
                loss_pose = loss_func_pose(predict, y_val)
                
                if normalize:
                    predict = normalize.denormalize(predict)
            
                pose_fact = PT.from_lie(pose)
                trajectory = TT.from_lie_relative(predict, pose_fact[0])
                
                loss_t = loss_func_trajectory(trajectory[:-1], pose_fact)
                
                
                loss = weigth_local * loss_pose + weigth_trajectory * loss_t
                
                path_lengh = trajectory.path_length()
                
                p_mean_loc = translation_rmse_drift(loss_func_pose.pos_loss, path_lengh)
                r_mean_loc = rotation_rmse_drift(loss_func_pose.r_loss, path_lengh)
                
                lm_count += 1
                loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
                loss_mean_test_pose = 1 / lm_count * loss_pose.item() + (1 - 1 / lm_count) * loss_mean_test_pose
                loss_mean_test_trajectory = 1 / lm_count * loss_t.item() + (1 - 1 / lm_count) * loss_mean_test_trajectory
                p_mean = 1 / lm_count * p_mean_loc.mean().item() + (1 - 1 / lm_count) * p_mean
                r_mean = 1 / lm_count * r_mean_loc.mean().item() + (1 - 1 / lm_count) * r_mean

                val_bar.set_postfix({
                'loss': loss_mean_test,
                'loss_pose': loss_mean_test_pose,
                'loss_trajectory': loss_mean_test_trajectory,
                'AT RMSE drift': p_mean,
                'AR RMSE drift': r_mean
                })

        dct_of_results['epoch'].append(epoch)
        dct_of_results['lose_train'].append(loss_mean_train),
        dct_of_results['lose_test'].append(loss_mean_test),
        dct_of_results['loss_pose_train'].append(loss_mean_train_pose)
        dct_of_results['loss_pose_test'].append(loss_mean_test_pose)
        dct_of_results['loss_trajectory_train'].append(loss_mean_train_trajectory)
        dct_of_results['loss_trajectory_test'].append(loss_mean_test_trajectory)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        with open(path_to_save_process_of_fitting, 'w', encoding='utf-8') as w:
            json.dump(dct_of_results, w, ensure_ascii=False, indent=4)
        
        # TODO Действительно ли это критерий отбора?
        # Выяснилось, что при отсутствии нормализации - Да
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results