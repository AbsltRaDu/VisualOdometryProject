import torch 
import numpy as np


class TorchImageToCvGray:
    '''
    Отдельный класс для применения в рамках Transform.Compose ТОЛЬКО ДЛЯ РАБОТЫ С КЛАССИК VO
    под структуру OpenCV
    
    По сути, просто костыль, чтобы не писать новый датасет
    '''

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        '''
        Преобразует формат изобаржений Dataset в формат, читаемый классическими поинт-детекторами
        '''
        
        img = img.detach().cpu() # Перегнали изображение на цп
        
        # img = img.permute(1, 2, 0)
        
        img = (img * 255.0).clip(0, 255).to(dtype=torch.uint8)
        
        
        
        if img.shape[0] == 3:
            img_graay = img[0, :, :].unsqueeze(0)
        else:
            raise ValueError(f'Неожиданное число каналов: {img.shape[0]}')
        
        return img_graay