import numpy as np 
import cv2

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
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
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

class TriangulationMod(Triangulation):
    
    def _get_focus(self):
        super()._get_focus()
        
        self._get_projection_matrices()
    
    def _get_projection_matrices(self):
        '''
        Создаёт матрицы проекции для левой и правой камеры.

        P_left: Левая камера считается началом координат stereo-системы.
        P_right: Правая камера сдвинута относительно левой на baseline.

        '''

        # Левая камера: R = I, t = 0
        Rt_left = np.hstack([
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64)
        ])

        # Правая камера: R = I, t = [-baseline, 0, 0]
        # Для стандартной rectified stereo часто используют именно -baseline.
        t_right = np.array([[-self.baseline], [0.0], [0.0]], dtype=np.float64)

        Rt_right = np.hstack([
            np.eye(3, dtype=np.float64),
            t_right
        ])

        # P = K @ [R | t]
        self.P_left = self.K @ Rt_left
        self.P_right = self.K @ Rt_right
        
    def points_3d_from_cv_triangulate(self, left: np.ndarray, right: np.ndarray):
        '''
        Восстанавливает 3D-точки через cv2.triangulatePoints.

        left: точки левого изображения формы (N, 2)
        right: соответствующие точки правого изображения формы (N, 2)
        '''

        # Приводим точки к float64
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)

        if left.ndim != 2 or left.shape[1] != 2:
            raise ValueError(f'left должен быть формы (N, 2), получил {left.shape}')

        if right.ndim != 2 or right.shape[1] != 2:
            raise ValueError(f'right должен быть формы (N, 2), получил {right.shape}')

        if len(left) != len(right):
            raise ValueError(
                f'left и right должны иметь одинаковую длину: '
                f'{len(left)} != {len(right)}'
            )

        
        pts_left = left.T
        pts_right = right.T

        
        points_4d = cv2.triangulatePoints(
            self.P_left,
            self.P_right,
            pts_left,
            pts_right
        )

        w = points_4d[3]
        valid_w = np.abs(w) > 1e-8

        points_3d = np.full((len(left), 3), np.nan, dtype=np.float64)
        points_3d[valid_w] = (points_4d[:3, valid_w] / w[valid_w]).T

        Z = points_3d[:, 2]

        valid = (valid_w & np.isfinite(points_3d).all(axis=1) & (Z > 0.0) & (Z < 100.0))

        return points_3d[valid].astype(np.float32), valid