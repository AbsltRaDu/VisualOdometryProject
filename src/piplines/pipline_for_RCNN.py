from collections import defaultdict

import torch
import torch.nn as nn 

from tqdm import tqdm

from src.metrics.KITTI_metrics import translation_rmse_drift, rotation_rmse_drift
from src.geometry.trigan import R_mat_to_euler_and_pose, euler_to_matrix_R, get_motion_matrix

def training(train_data, test_data, model, loss_func, optimizer, epochs, device, name_of_model, squueze=False):
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