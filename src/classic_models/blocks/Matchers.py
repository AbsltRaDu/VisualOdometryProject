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