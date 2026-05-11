from typing import Optional

import torch
import torch.nn as nn 
import numpy as np
import cv2

from src.classic_models.blocks.Matchers import Matcher, TemporalMatcher
from src.geometry.Triangulation import Triangulation
from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.RotationTorch import RotationTorch as RT

class FeaturesMethod(nn.Module):
    
    def __init__(self, width: int, height: int,  new_width: int, new_height: int, fov_deg: float, baseline: float, detection_algoritm, matcher, k_match: int = 2, 
                 min_stereo_matches: int = 20, min_3d_points: int = 20, min_pnp_points: int = 20, min_pnp_inliers: int = 10, return_debug: bool = False,):
        '''
        width, height - ширина, высота изображения в пикселях
        fov_deg - угол обзора камеры (в градусах)
        baseline - расстояние между камерами
        detection_algoritm - алгоритм детекции и дескрипизации точек
        matcher - мэтчер
        k_match - кол-во ближайших соседей для мэтчера
        '''
        
        
        super().__init__()
        self.width = width
        self.height = height
        self.new_width = new_width
        self.new_height = new_height
        self.fov_deg = fov_deg
        self.baseline = baseline
        
        self.detection_algoritm = detection_algoritm
        self.matcher = Matcher(matcher=matcher, k=k_match)
        self.temporal_matcher = TemporalMatcher(matcher=matcher, k=k_match)
        self.triangulation = Triangulation(self.width, self.height, self.new_width, self.new_height, self.fov_deg, self.baseline)
        
        self.min_stereo_matches = min_stereo_matches
        self.min_3d_points = min_3d_points
        self.min_pnp_points = min_pnp_points
        self.min_pnp_inliers = min_pnp_inliers
        self.return_debug = return_debug
    
    def _zero_prediction(self, reason: str, debug: Optional[dict]):
        '''
        Возвращает нулевое движение
        '''
        
        if debug is None:
            debug = {}

        # Нулевой вектор поворота.
        rvec_pred = torch.zeros(3, dtype=torch.float64)

        # Нулевой вектор смещения.
        tvec_pred = torch.zeros(3, dtype=torch.float64)

        # Собираем PoseTorch и переводим в Lie-вектор.
        rvec_pred = RT.from_rotvec(rvec_pred)
        y_pred = PT.from_rt(rvec_pred, tvec_pred).as_lie()

        debug['success'] = False
        debug['reason'] = reason

        if self.return_debug:
            return y_pred, debug

        return y_pred
        
    
    def forward(self, img_left_1, img_right_1, img_left_2):
        
        with torch.no_grad(): # Градиенты тут нафиг не нужны, ничего не обучается ведь
            
            debug = {
                'success': False,
                'reason': None,
                'num_kp_left_1': 0,
                'num_kp_right_1': 0,
                'num_kp_left_2': 0,
                'num_stereo_matches': 0,
                'num_3d_points': 0,
                'num_pnp_points': 0,
                'num_pnp_inliers': 0,
            }
            
            # Блок детекции и дескрипизации
            kp1, des1 = self.detection_algoritm.detectAndCompute(img_left_1, None)
            kp3, des3 = self.detection_algoritm.detectAndCompute(img_left_2, None)
            kp2, des2 = self.detection_algoritm.detectAndCompute(img_right_1, None)
            
            debug['num_kp_left_1'] = 0 if kp1 is None else len(kp1)
            debug['num_kp_right_1'] = 0 if kp2 is None else len(kp2)
            debug['num_kp_left_2'] = 0 if kp3 is None else len(kp3)
            
            if des1 is None or des2 is None or des3 is None:
                return self._zero_prediction(
                    reason='Нет дескриптора',
                    debug=debug,
                )
                
            if len(kp1) == 0 or len(kp2) == 0 or len(kp3) == 0:
                return self._zero_prediction(
                    reason='Нет ключевых точек',
                    debug=debug,
                )
            
            # Блок мэтчинга
            good_matches = self.matcher(des1, des2)
            
            debug['num_stereo_matches'] = len(good_matches)
            if len(good_matches) < self.min_stereo_matches:
                return self._zero_prediction(
                    reason='Недостаточно найденных совпадений на стерео-кадрах',
                    debug=debug,
                )
            
            pts_left = np.float32([
                kp1[m.queryIdx].pt
                for m in good_matches
            ])
            left_indeces = np.asarray([m.queryIdx for m in good_matches], dtype=np.int32)
            pts_right = np.float32([
                kp2[m.trainIdx].pt
                for m in good_matches
            ])
            
            # Блок триангуляции
            points_3d, valid = self.triangulation.points_3d_from_matches(pts_left, pts_right)
            
            debug['num_3d_points'] = len(points_3d)
            if len(points_3d) < self.min_3d_points:
                return self._zero_prediction(
                    reason='Недостаточно 3D точек',
                    debug=debug,
                )
            
            # Блок мэтча t и t+1
            indeces_3d_point_left1 = left_indeces[valid]
            left_idx_to_3d_idx = {
                int(left_idx): i
                for i, left_idx in enumerate(indeces_3d_point_left1)
            }
            object_points, image_points = self.temporal_matcher(des1, des3, kp3, points_3d, left_idx_to_3d_idx)
            
            debug['num_pnp_points'] = len(object_points)
            if len(object_points) < self.min_pnp_points:
                return self._zero_prediction(
                    reason='Недостаточно найденных точек в результате PnP',
                    debug=debug,
                )

            # Блок расчета сдвига
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.triangulation.K,
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if inliers is not None:
                debug['num_pnp_inliers'] = len(inliers)
            if not success:
                return self._zero_prediction(
                    reason='pnp_success_false',
                    debug=debug,
                )
            if rvec is None or tvec is None:
                return self._zero_prediction(
                    reason='pnp_empty_pose',
                    debug=debug,
                )
            if inliers is None or len(inliers) < self.min_pnp_inliers:
                return self._zero_prediction(
                    reason='not_enough_pnp_inliers',
                    debug=debug,
                )
            
            rvec_pred = torch.from_numpy(rvec.squeeze())
            tvec_pred = torch.from_numpy(tvec.squeeze())
            
            rvec_pred = RT.from_rotvec(rvec_pred)
            y_pred = PT.from_rt(rvec_pred, tvec_pred).as_lie()
            
            debug["success"] = True
            debug["reason"] = "ok"

            if self.return_debug:
                return y_pred, debug

            return y_pred