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
        
    def disparity_and_depth(self, left: np.ndarray, right: np.ndarray, disparity_map: np.ndarray = None):
        '''
        Восстанавливает глубину по стерео-точкам, возвращает диспаритет
        '''
        if disparity_map is None:
            disparity = left[..., 0] - right[..., 0]
            valid = disparity > 1e-6 # TODO продумать вариант гибкого фильтра
        else:
            h, w = disparity_map.shape

            x_i = np.round(left[:, 0]).astype(np.int32)
            y_i = np.round(left[:, 1]).astype(np.int32)

            # Проверяем, что keypoints попали внутрь изображения
            inside = (
                (x_i >= 0) & (x_i < w) &
                (y_i >= 0) & (y_i < h)
            )

            disparity = np.full(len(left), np.nan, dtype=np.float32)

            # Берём disparity из карты только для валидных пикселей
            disparity[inside] = disparity_map[y_i[inside], x_i[inside]]
            valid = inside & (disparity > 1e-1)
        
        

        Z = np.full(len(left), fill_value=np.nan, dtype=np.float64)
        
        Z[valid] = self.fx * self.baseline / disparity[valid]
        
        return Z, disparity, valid
    
    def points_3d_from_matches(self, left: np.ndarray, right: np.ndarray, disparity: np.ndarray = None):
        '''
        Восстанавливает 3D точки по матчу стерео-точкам
        '''
        
        Z, disparity, valid = self.disparity_and_depth(left, right, disparity)
        valid = valid & np.isfinite(Z) & (Z > 0.0) & (Z < 100.0)

        # Берём только валидные точки
        x = left[..., 0][valid]
        y = left[..., 1][valid]
        z = Z[valid]

        # Обратная проекция пикселя в 3D.
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy

        points_3d = np.stack([X, Y, z], axis=1).astype(np.float32)

        return points_3d, valid