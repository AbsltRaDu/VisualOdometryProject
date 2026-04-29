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

def training_CNN_JointTraning(train_data, test_data, model, loss_func_pose: PoseLoss, loss_func_trajectory: PoseLossTrajectory, optimizer, epochs, device, name_of_model, path_to_save_process_of_fitting,
                              squueze=False, normalize=None, weigth_local=1000, weigth_trajectory=1):
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
        
        if epoch % 5 == 0:
            weigth_local -= 50
            weigth_trajectory += 50
        
        
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

def training_RCNN(train_data, test_data, model, loss_func, optimizer, epochs, device, name_of_model, squueze=False):
    best_score = 10**10
    dct_of_results = defaultdict(list)
    
    for epoch in range(epochs):
    
        loss_mean_train = 0
        loss_mean_test = 0
        r_mean, p_mean = 0, 0
        path_lengh = 0
        lm_count = 0
  
        train_bar = tqdm(train_data, desc=f'Эпоха тренировочная {epoch+1}/{epochs}', position=0)
        
        model.train()
        
        for x_train, y_train, T_m in train_bar:
            x_train = x_train.to(device) 
            y_train = y_train.to(device) 
            
            predict = model(x_train)
            predict = predict.unsqueeze(0) if squueze else predict
            
            
            loss = loss_func(predict, y_train)
            lm_count += 1
            loss_mean_train = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_train
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix({
                'loss': loss_mean_train
            })
            
        model.eval()
        
        val_bar = tqdm(test_data, desc=f'Эпоха валидационная {epoch+1}/{epochs}', position=1)
        
        lm_count = 0
        
        for x_val, y_val, _ in val_bar:
            x_val = x_val.to(device) 
            y_val = y_val.to(device) 
            
            with torch.no_grad():
                predict = model(x_val)
                predict = predict.unsqueeze(0) if squueze else predict
                loss = loss_func(predict, y_val)
                
                path_lengh += torch.linalg.norm(y_val[:, :3], dim=1).sum() # Длина пути, пройденного в батче
                
                p_mean_loc = translation_rmse_drift(predict[:, :3], y_val[:, :3], path_lengh)
                r_mean_loc = rotation_rmse_drift(predict[:, 3:], y_val[:, 3:], path_lengh)
                
                lm_count += 1
                loss_mean_test = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean_test
                p_mean = 1 / lm_count * p_mean_loc.mean().item() + (1 - 1 / lm_count) * p_mean
                r_mean = 1 / lm_count * r_mean_loc.mean().item() + (1 - 1 / lm_count) * r_mean

                val_bar.set_postfix({
                'loss': loss_mean_test,
                'Average Translational RMSE drift': p_mean,
                'Average Rotational RMSE drift': r_mean,
                'path_lenght': path_lengh
                })

        
        dct_of_results['loss_train'].append(loss_mean_train)
        dct_of_results['loss_test'].append(loss_mean_test)
        dct_of_results['Average Translational RMSE drift'].append(p_mean)
        dct_of_results['Average Rotational RMSE drift'].append(r_mean)
        
        if p_mean + r_mean <= best_score:
            best_score = p_mean + r_mean
            torch.save(model.state_dict(), name_of_model)
            print('Модель сохранена')
            
    return dct_of_results