import torch
from tqdm import tqdm

def get_tensors_of_x_y(train_data, model_cnn, device):

    lst_of_y = []
    lst_of_x = []
    lst_of_T = []
    
    model_cnn.eval()
    with torch.no_grad():
        train_bar = tqdm(train_data, desc=f'Эпоха формирования базы представлений CNN', position=0)
        for x_train, y_train, T_m in train_bar:
            x = model_cnn(x_train.to(device)) # (B, C)
            lst_of_x.append(x.cpu())
            lst_of_y.append(y_train.cpu())
            lst_of_T.append(T_m.cpu())
            
        tensor_of_x = torch.cat(lst_of_x, dim=0) # Тензор представлений
        tensor_of_y = torch.cat(lst_of_y, dim=0) # Тензор представлений
        tensor_of_T = torch.cat(lst_of_T, dim=0)
        
    return tensor_of_x, tensor_of_y, tensor_of_T