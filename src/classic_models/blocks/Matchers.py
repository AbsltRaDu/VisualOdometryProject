import cv2
import numpy as np

class Matcher:
    
    def __init__(self, matcher: cv2.BFMatcher, k: int=2):
        self.k = k
        
        
        if isinstance(matcher, cv2.BFMatcher):
            self.matcher = matcher
            
        else:
            raise TypeError(f'Объект matcher должен принадлежать классу cv2.BFMatcher')
        
    def __call__(self, des1, des3):
        knn_matcher = self.matcher.knnMatch(des1, des3, k=self.k)

        good_matches = []

        for pair in knn_matcher:
            
            # Если меньше соседей
            if len(pair) < self.k:
                continue

            m, n = pair
            
            # Отсеиваем ненадежные соответствия
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches
    
class TemporalMatcher(Matcher):
    
    __matcher_call = Matcher.__call__
    
    def __init__(self, matcher: cv2.BFMatcher, k: int=2):
        super().__init__(matcher, k)
    
    def __call__(self, des1, des2, kp2, points_3d, indeces_3d_point_left1):
        '''
        des1, des2: дескрипизация точек на t1 и t2 кадрах
        kp2: 2D координаты точек t2
        points_3d: 3D координаты точек t1
        indeces_3d_point_left1: индексы 3D точек, которые прошли триангуляцию
        '''
        
        t_matches = self.__matcher_call(des1, des2) # Выполнили мэтчинг, применив метод родителя
        
        object_points = []
        image_points = []

        for m in t_matches:

            idx_left_t = int(m.queryIdx)
            idx_left_t1 = int(m.trainIdx)

            if idx_left_t not in indeces_3d_point_left1:
                continue

            # Находим соответствующую 3D-точку
            idx_3d = indeces_3d_point_left1[idx_left_t]
            X_3d = points_3d[idx_3d]

            # Берём 2D-координату этой же точки на кадре t+1
            x_2d, y_2d = kp2[idx_left_t1].pt

            object_points.append(X_3d)
            image_points.append([x_2d, y_2d])

        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        
        
        return object_points, image_points
    
class StereoFilter:
    
    def __init__(self, max_dy: int = 2, min_disparity: int = 1, max_disparity: int = 120):
        self.max_dy = max_dy
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        
    def __call__(self, stereo_matches, kp1_left_1, kp2_right_1):
        
        filtered_matches = []

        for m in stereo_matches:

            x_l, y_l = kp1_left_1[m.queryIdx].pt
            x_r, y_r = kp2_right_1[m.trainIdx].pt

            dy = abs(y_l - y_r)

            disparity = x_l - x_r

            if dy > 2.0:
                continue

            if disparity <= 1.0:
                continue

            if disparity > 120.0:
                continue

            filtered_matches.append(m)
            
        return filtered_matches
    
class FundamentalMat:
    
    def __init__(self, method=cv2.FM_RANSAC, ransacReprojThreshold: float = 1.0, confidence: float = 0.99, min_points: int = 8):
        self.method = method
        self.ransacReprojThreshold = ransacReprojThreshold
        self.confidence = confidence
        self.min_points = min_points
    
    def __call__(self, good_matches, pts_left, pts_right, debug=None):
        
        if debug is None:
            debug = {}

        pts_left = np.asarray(pts_left, dtype=np.float32)
        pts_right = np.asarray(pts_right, dtype=np.float32)

        debug['num_fm_input_matches'] = len(good_matches)
        debug['num_fm_inliers'] = 0
        debug['fm_success'] = False
        debug['fm_reason'] = None

        if len(good_matches) == 0:
            debug['fm_reason'] = 'Нет мэтчей'
            return good_matches, pts_left, pts_right, None, debug

        if pts_left.ndim != 2 or pts_left.shape[1] != 2:
            debug['fm_reason'] = f'bad_pts_left_shape_{pts_left.shape}'
            return good_matches, pts_left, pts_right, None, debug

        if pts_right.ndim != 2 or pts_right.shape[1] != 2:
            debug['fm_reason'] = f'bad_pts_right_shape_{pts_right.shape}'
            return good_matches, pts_left, pts_right, None, debug

        if len(good_matches) != len(pts_left) or len(good_matches) != len(pts_right):
            debug['fm_reason'] = (
                f'length_mismatch_matches_{len(good_matches)}_'
                f'left_{len(pts_left)}_right_{len(pts_right)}'
            )
            return good_matches, pts_left, pts_right, None, debug

        if len(good_matches) < self.min_points:
            debug['fm_reason'] = 'Недостаточно точек для фильтрации'
            return good_matches, pts_left, pts_right, None, debug
        

        try:
            F, mask = cv2.findFundamentalMat(
                pts_left,
                pts_right,
                method=self.method,
                ransacReprojThreshold=self.ransacReprojThreshold,
                confidence=self.confidence,
            )
        except cv2.error as e:
            debug["fm_reason"] = "cv2_error"
            debug["fm_cv2_error"] = str(e)
            return good_matches, pts_left, pts_right, None, debug
        
        if F is None or mask is None:
            debug['fm_reason'] = 'FM пустая'
            return good_matches, pts_left, pts_right, None, debug

        mask = mask.ravel().astype(bool)

        if len(mask) != len(good_matches):
            debug['fm_reason'] = f'bad_mask_length_{len(mask)}'
            return good_matches, pts_left, pts_right, None, debug

        filtered_matches = [
            m for m, keep in zip(good_matches, mask)
            if keep
        ]

        filtered_pts_left = pts_left[mask]
        filtered_pts_right = pts_right[mask]

        debug['num_fm_inliers'] = len(filtered_matches)
        debug['fm_success'] = True
        debug['fm_reason'] = 'ok'

        return filtered_matches, filtered_pts_left, filtered_pts_right, mask, debug