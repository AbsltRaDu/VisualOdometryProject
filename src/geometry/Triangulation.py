import numpy as np 

class Triangulation:
    '''
    NUMPY реализация
    
    Класс восстановления глубины по данным о точках со стереокамер
    '''
    
    def __init__(self, width, height, new_width, new_height, fov_deg, baseline):
        '''
        width, height - ширина, высота изображения в пикселях
        fov_deg - угол обзора камеры (в градусах)
        baseline - расстояние между камерами
        '''
        
        self.width = width
        self.height = height
        self.new_width = new_width
        self.new_height = new_height
        self.fov_deg = np.deg2rad(fov_deg)
        self.baseline = baseline
        
        self._get_focus()

    def _get_focus(self):
        fx_orig = self.width / (2 * np.tan(self.fov_deg / 2))

        fy_orig = fx_orig # TODO нужно сделать обработку для НЕ квадратных пикселей 
        
        # TODO продумать обработку для неидеальных камер. Алгоритм писался для синтетики, поэтому тут центр ровный
        cx_orig = self.width / 2
        cy_orig = self.height / 2
        
        scale_x = self.new_width / self.width
        scale_y = self.new_height / self.height
        
        # Масштабируем 
        self.fx = fx_orig * scale_x
        self.fy = fy_orig * scale_y
        self.cx = cx_orig * scale_x
        self.cy = cy_orig * scale_y
        
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        
    def disparity_and_depth(self, left: np.ndarray, right: np.ndarray):
        '''
        Восстанавливает глубину по стерео-точкам, возвращает диспаритет
        '''
        
        disparity = left[..., 0] - right[..., 0]
        
        # print("disparity min:", disparity.min())
        # print("disparity mean:", disparity.mean())
        # print("disparity max:", disparity.max())
        
        valid = disparity > 1e-6 # TODO продумать вариант гибкого фильтра

        Z = np.full_like(disparity, fill_value=np.nan, dtype=np.float64)
        
        # print("Z min:", np.nanmin(Z))
        # print("Z mean:", np.nanmean(Z))
        # print("Z max:", np.nanmax(Z))
        
        Z[valid] = self.fx * self.baseline / disparity[valid]
        
        return Z, disparity, valid
    
    def points_3d_from_matches(self, left: np.ndarray, right: np.ndarray):
        '''
        Восстанавливает 3D точки по матчу стерео-точкам
        '''
        
        Z, disparity, valid = self.disparity_and_depth(left, right)

        # Берём только валидные точки
        x = left[..., 0][valid]
        y = left[..., 1][valid]
        z = Z[valid]

        # Обратная проекция пикселя в 3D.
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy

        points_3d = np.stack([X, Y, z], axis=1).astype(np.float32)

        return points_3d, valid