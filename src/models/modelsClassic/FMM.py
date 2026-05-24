from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.RotationTorch import RotationTorch as RT


FeatureType = Literal["sift", "orb"]


@dataclass
class StereoCameraConfig:
    '''
    Минимальная калибровка стереокамеры.

    width, height: исходный размер изображения.

    new_width, new_height: размер изображения после resize. Если resize нет, равны width/height.

    fov_deg: горизонтальный угол обзора в градусах.

    baseline: расстояние между камерами в метрах.
    '''

    width: int = 752
    height: int = 480
    new_width: int = 752
    new_height: int = 480
    fov_deg: float = 90.0
    baseline: float = 0.2

    def make_K(self) -> np.ndarray:
        '''
        Создаёт матрицу камеры K с учётом масштабирования.
        '''


        fov_rad = np.deg2rad(self.fov_deg)
        fx_orig = self.width / (2.0 * np.tan(fov_rad / 2.0))
        fy_orig = fx_orig


        cx_orig = self.width / 2.0
        cy_orig = self.height / 2.0

        scale_x = self.new_width / self.width
        scale_y = self.new_height / self.height
        
        fx = fx_orig * scale_x
        fy = fy_orig * scale_y
        cx = cx_orig * scale_x
        cy = cy_orig * scale_y

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return K

    def make_projection_matrices(self, right_sign: float = -1.0):
        '''
        Создаёт P_left и P_right для cv2.triangulatePoints.

        P_left = K [I | 0]
        P_right = K [I | t]

        right_sign: знак baseline для правой камеры
        '''

        K = self.make_K()


        Rt_left = np.hstack(
            [
                np.eye(3, dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
            ]
        )

        t_right = np.array(
            [[right_sign * self.baseline], [0.0], [0.0]],
            dtype=np.float64,
        )

        Rt_right = np.hstack([np.eye(3, dtype=np.float64), t_right])

        P_left = K @ Rt_left
        P_right = K @ Rt_right

        return K, P_left, P_right


@dataclass
class FeatureStereoVOConfig:
    '''
    Конфигурация SIFT/ORB stereo VO.
    '''

    camera: StereoCameraConfig

    feature_type: FeatureType = "sift"
    nfeatures: int = 8000

    ratio_test: float = 0.75

    max_stereo_dy: float = 2.0
    min_disparity: float = 1.0
    max_disparity: float = 256.0

    min_depth: float = 0.2
    max_depth: float = 100.0

    min_stereo_matches: int = 20
    min_3d_points: int = 20
    min_temporal_matches: int = 20
    min_pnp_points: int = 20
    min_pnp_inliers: int = 10

    pnp_iterations: int = 100
    pnp_reprojection_error: float = 3.0
    pnp_confidence: float = 0.999

    use_stereo_fundamental: bool = False
    fundamental_threshold: float = 1.0
    fundamental_confidence: float = 0.99

    right_projection_sign: float = -1.0

    return_debug: bool = False


class CV2FeatureStereoVO(nn.Module):

    def __init__(self, config: FeatureStereoVOConfig):
        super().__init__()

        self.config = config

        self.K, self.P_left, self.P_right = config.camera.make_projection_matrices(
            right_sign=config.right_projection_sign
        )
        
        self.dist = None
        
        self.detector, self.matcher = self._build_feature_backend()

    def _build_feature_backend(self):
        '''
        Создаёт SIFT/ORB детектор и подходящий BFMatcher.
        '''

        if self.config.feature_type == "sift":
            detector = cv2.SIFT_create(nfeatures=self.config.nfeatures)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            return detector, matcher

        if self.config.feature_type == "orb":
            detector = cv2.ORB_create(
                nfeatures=self.config.nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                patchSize=31,
                fastThreshold=7,
            )
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            return detector, matcher

        raise ValueError(f"Неизвестный feature_type: {self.config.feature_type}")


    def _zero_prediction(self, reason: str, debug: Optional[dict] = None):
        '''
        Возвращает нулевое движение в слуаче неудачи.
        '''

        if debug is None:
            debug = {}

        rvec = torch.zeros(3, dtype=torch.float64)
        tvec = torch.zeros(3, dtype=torch.float64)
        R = RT.from_rotvec(rvec)
        y_pred = PT.from_rt(R, tvec).as_lie()

        debug["success"] = False
        debug["reason"] = reason

        if self.config.return_debug:
            return y_pred, debug
        return y_pred

    def _detect(self, img: np.ndarray):
        '''
        Детекция keypoints и дескрипторов.
        '''

        kp, des = self.detector.detectAndCompute(img, None)
        return kp, des

    def _knn_ratio_match(self, des_a, des_b):
        '''
        KNN мэтчинг.
        '''

        if des_a is None or des_b is None:
            return []

        if len(des_a) == 0 or len(des_b) == 0:
            return []

        try:
            pairs = self.matcher.knnMatch(des_a, des_b, k=2)
        except cv2.error:
            return []

        good = []

        for pair in pairs:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < self.config.ratio_test * n.distance:
                good.append(m)

        return good

    def _stereo_filter(self, matches, kp_left, kp_right):
        '''
        Фильтрует stereo matches.
        '''

        filtered = []

        for m in matches:
            x_l, y_l = kp_left[m.queryIdx].pt
            x_r, y_r = kp_right[m.trainIdx].pt

            dy = abs(y_l - y_r)
            disparity = x_l - x_r

            if dy > self.config.max_stereo_dy:
                continue

            if disparity <= self.config.min_disparity:
                continue

            if disparity >= self.config.max_disparity:
                continue

            filtered.append(m)

        return filtered

    def _fundamental_filter(self, matches, kp_left, kp_right, debug):
        '''
        Опциональный Fundamental Matrix RANSAC
        '''

        if len(matches) < 8:
            debug["fundamental_reason"] = "not_enough_matches"
            return matches

        pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
        pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches])

        try:
            F, mask = cv2.findFundamentalMat(
                pts_left,
                pts_right,
                cv2.FM_RANSAC,
                self.config.fundamental_threshold,
                self.config.fundamental_confidence,
            )
        except cv2.error as e:
            debug["fundamental_reason"] = "cv2_error"
            debug["fundamental_cv2_error"] = str(e)
            return matches

        if F is None or mask is None:
            debug["fundamental_reason"] = "F_or_mask_is_none"
            return matches

        mask = mask.ravel().astype(bool)

        if len(mask) != len(matches):
            debug["fundamental_reason"] = "bad_mask_length"
            return matches

        filtered = [m for m, keep in zip(matches, mask) if keep]
        debug["num_stereo_matches_after_fundamental"] = len(filtered)
        debug["fundamental_reason"] = "ok"

        return filtered

    def _triangulate_cv2(self, matches, kp_left, kp_right):
        '''
        Решение задачи триангуляции
        '''

        if len(matches) == 0:
            return np.empty((0, 3), dtype=np.float32), {}

        pts_left = np.float64([kp_left[m.queryIdx].pt for m in matches])
        pts_right = np.float64([kp_right[m.trainIdx].pt for m in matches])

        # cv2.triangulatePoints ждёт точки формы (2, N).
        points_4d = cv2.triangulatePoints(
            self.P_left,
            self.P_right,
            pts_left.T,
            pts_right.T,
        )

        w = points_4d[3]
        valid_w = np.abs(w) > 1e-8

        points_3d_all = np.full((len(matches), 3), np.nan, dtype=np.float64)
        points_3d_all[valid_w] = (points_4d[:3, valid_w] / w[valid_w]).T

        Z = points_3d_all[:, 2]

        valid = (
            valid_w
            & np.isfinite(points_3d_all).all(axis=1)
            & (Z > self.config.min_depth)
            & (Z < self.config.max_depth)
        )

        points_3d = points_3d_all[valid].astype(np.float32)

        left_idx_to_3d_idx = {}
        valid_counter = 0

        for match_idx, m in enumerate(matches):
            if not valid[match_idx]:
                continue

            left_idx_to_3d_idx[int(m.queryIdx)] = valid_counter
            valid_counter += 1

        return points_3d, left_idx_to_3d_idx

    def _build_3d_2d_correspondences(self, temporal_matches, kp_left_2, points_3d, left_idx_to_3d_idx):
        '''
        Построение пар точек 3D-2D.
        '''

        object_points = []
        image_points = []

        for m in temporal_matches:
            idx_left_t = int(m.queryIdx)
            idx_left_t1 = int(m.trainIdx)

            if idx_left_t not in left_idx_to_3d_idx:
                continue

            idx_3d = left_idx_to_3d_idx[idx_left_t]
            X = points_3d[idx_3d]
            x2, y2 = kp_left_2[idx_left_t1].pt

            object_points.append(X)
            image_points.append([x2, y2])

        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
        )

    def _pose_from_pnp(self, rvec, tvec, debug):
        '''
        Переводит OpenCV PnP результат в Ли-вектор.
        '''

        rvec_t = torch.from_numpy(rvec.squeeze()).to(dtype=torch.float64)
        tvec_t = torch.from_numpy(tvec.squeeze()).to(dtype=torch.float64)

        R = RT.from_rotvec(rvec_t)

        pose = PT.from_rt(R, tvec_t).inv()

        C = torch.tensor([
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ], dtype=pose.dtype, device=pose.device)
        pose = pose.change_basis(C)

        y_pred = pose.as_lie()

        debug["success"] = True
        debug["reason"] = "ok"
        debug["pnp_raw_tvec"] = tvec.squeeze().tolist()
        debug["pnp_motion_t"] = pose.t.detach().cpu().tolist()

        return y_pred, debug

    def forward(self, img_left_1, img_right_1, img_left_2):

        with torch.no_grad():
            debug = {
                "success": False,
                "reason": None,
                "feature_type": self.config.feature_type,
                "num_kp_left_1": 0,
                "num_kp_right_1": 0,
                "num_kp_left_2": 0,
                "num_stereo_matches_raw": 0,
                "num_stereo_matches_filtered": 0,
                "num_3d_points": 0,
                "num_temporal_matches": 0,
                "num_pnp_points": 0,
                "num_pnp_inliers": 0,
            }

            # Детекция и дескрипторы.
            kp_left_1, des_left_1 = self._detect(img_left_1)
            kp_right_1, des_right_1 = self._detect(img_right_1)
            kp_left_2, des_left_2 = self._detect(img_left_2)

            debug["num_kp_left_1"] = 0 if kp_left_1 is None else len(kp_left_1)
            debug["num_kp_right_1"] = 0 if kp_right_1 is None else len(kp_right_1)
            debug["num_kp_left_2"] = 0 if kp_left_2 is None else len(kp_left_2)

            if des_left_1 is None or des_right_1 is None or des_left_2 is None:
                return self._zero_prediction("no_descriptors", debug)

            # Stereo matching left_t <-> right_t.
            stereo_matches = self._knn_ratio_match(des_left_1, des_right_1)
            debug["num_stereo_matches_raw"] = len(stereo_matches)

            if len(stereo_matches) < self.config.min_stereo_matches:
                return self._zero_prediction("not_enough_raw_stereo_matches", debug)

            # Простая stereo-фильтрация.
            stereo_matches = self._stereo_filter(stereo_matches, kp_left_1, kp_right_1)
            debug["num_stereo_matches_filtered"] = len(stereo_matches)

            if len(stereo_matches) < self.config.min_stereo_matches:
                return self._zero_prediction("not_enough_filtered_stereo_matches", debug)

            # Опциональный Fundamental Matrix filter.
            if self.config.use_stereo_fundamental:
                stereo_matches = self._fundamental_filter(stereo_matches, kp_left_1, kp_right_1, debug)

                if len(stereo_matches) < self.config.min_stereo_matches:
                    return self._zero_prediction("not_enough_stereo_matches_after_fundamental", debug)

            # Триангуляция через cv2.triangulatePoints.
            points_3d, left_idx_to_3d_idx = self._triangulate_cv2(
                stereo_matches,
                kp_left_1,
                kp_right_1,
            )

            debug["num_3d_points"] = int(len(points_3d))

            if len(points_3d) < self.config.min_3d_points:
                return self._zero_prediction("not_enough_3d_points", debug)

            # Temporal matching left_t <-> left_t+1.
            temporal_matches = self._knn_ratio_match(des_left_1, des_left_2)
            debug["num_temporal_matches"] = len(temporal_matches)

            if len(temporal_matches) < self.config.min_temporal_matches:
                return self._zero_prediction("not_enough_temporal_matches", debug)

            # 3D_t  с 2D_t+1.
            object_points, image_points = self._build_3d_2d_correspondences(
                temporal_matches,
                kp_left_2,
                points_3d,
                left_idx_to_3d_idx,
            )

            debug["num_pnp_points"] = int(len(object_points))

            if len(object_points) < self.config.min_pnp_points:
                return self._zero_prediction("not_enough_pnp_points", debug)

            # PnP RANSAC.
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=self.K,
                    distCoeffs=self.dist,
                    iterationsCount=self.config.pnp_iterations,
                    reprojectionError=self.config.pnp_reprojection_error,
                    confidence=self.config.pnp_confidence,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error as e:
                debug["cv2_error"] = str(e)
                return self._zero_prediction("pnp_cv2_error", debug)

            if inliers is not None:
                debug["num_pnp_inliers"] = int(len(inliers))

            if not success:
                return self._zero_prediction("pnp_success_false", debug)

            if rvec is None or tvec is None:
                return self._zero_prediction("pnp_empty_pose", debug)

            if inliers is None or len(inliers) < self.config.min_pnp_inliers:
                return self._zero_prediction("not_enough_pnp_inliers", debug)

            y_pred, debug = self._pose_from_pnp(rvec, tvec, debug)

            if self.config.return_debug:
                return y_pred, debug

            return y_pred


