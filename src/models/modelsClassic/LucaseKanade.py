from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.geometry.PoseTorch import PoseTorch as PT
from src.geometry.RotationTorch import RotationTorch as RT
from src.geometry.Triangulation import TriangulationMod


class LKOpticalFlowVO(nn.Module):
    """
    Оптический поток Лукаса-Канаде
    """

    def __init__(
        self,
        width: int,
        height: int,
        new_width: int,
        new_height: int,
        fov_deg: float,
        baseline: float,
        max_corners: int = 3000,
        quality_level: float = 0.01,
        min_distance: int = 7,
        block_size: int = 7,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_max_error: float = 30.0,
        fb_max_error: float = 1.5,
        min_depth: float = 0.2,
        max_depth: float = 100.0,
        min_pnp_points: int = 20,
        min_pnp_inliers: int = 10,
        return_debug: bool = False,
    ):
        """
        width, height:
            исходный размер изображения камеры до resize.

        new_width, new_height:
            размер изображения после resize, который реально приходит в модель.

        fov_deg:
            горизонтальный угол обзора камеры в градусах.

        baseline:
            расстояние между камерами в метрах.

        max_corners:
            максимальное число точек, которые ищем на left_t.

        quality_level:
            порог качества углов для cv2.goodFeaturesToTrack.
            Меньше значение -> больше точек, но больше шума.

        min_distance:
            минимальная дистанция между найденными точками в пикселях.

        block_size:
            размер окна для оценки качества углов.

        lk_win_size:
            размер окна Lucas–Kanade optical flow.

        lk_max_level:
            число уровней пирамиды LK.
            Больше -> лучше отслеживание больших смещений, но медленнее.

        lk_max_error:
            максимальная ошибка LK по status/error.

        fb_max_error:
            forward-backward check.
            Точка отслеживается t -> t+1, потом обратно t+1 -> t.
            Если вернулась далеко от исходной, точку выбрасываем.

        min_depth, max_depth:
            допустимый диапазон глубины.

        min_pnp_points:
            минимум 3D->2D соответствий перед PnP.

        min_pnp_inliers:
            минимум inliers после PnP RANSAC.
        """
        super().__init__()

        self.width = width
        self.height = height
        self.new_width = new_width
        self.new_height = new_height
        self.fov_deg = fov_deg
        self.baseline = baseline

        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.lk_max_error = lk_max_error
        self.fb_max_error = fb_max_error

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_pnp_points = min_pnp_points
        self.min_pnp_inliers = min_pnp_inliers
        self.return_debug = return_debug

        self.triangulation = TriangulationMod(
            self.width,
            self.height,
            self.new_width,
            self.new_height,
            self.fov_deg,
            self.baseline,
        )

        block = 7
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,
            blockSize=block,
            P1=8 * 1 * block ** 2,
            P2=32 * 1 * block ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _zero_prediction(self, reason: str, debug: Optional[dict] = None):
        """
        Возвращает нулевое движение, если pipeline не смог оценить позу.
        """
        if debug is None:
            debug = {}

        # Нулевой rotvec и нулевая трансляция.
        rvec = torch.zeros(3, dtype=torch.float64)
        tvec = torch.zeros(3, dtype=torch.float64)

        # Собираем PoseTorch и переводим в Lie-вектор проекта.
        R = RT.from_rotvec(rvec)
        y_pred = PT.from_rt(R, tvec).as_lie()

        debug["success"] = False
        debug["reason"] = reason

        if self.return_debug:
            return y_pred, debug
        return y_pred

    def _detect_points(self, img_left_1: np.ndarray):
        """
        Используем cv2.goodFeaturesToTrack детектор углов Shi–Tomasi
        """
        pts = cv2.goodFeaturesToTrack(
            image=img_left_1,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

        # OpenCV возвращает форму (N, 1, 2), если точки найдены.
        return pts

    def _track_lk(self, img_left_1: np.ndarray, img_left_2: np.ndarray, pts_1: np.ndarray):
        """
        Отслеживает точки pts_1 из left_t в left_t+1 через Lucas–Kanade.
        """
        lk_params = dict(
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        pts_2, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            img_left_1,
            img_left_2,
            pts_1,
            None,
            **lk_params,
        )

        if pts_2 is None or status_fwd is None:
            return None, None, None

        pts_1_back, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(
            img_left_2,
            img_left_1,
            pts_2,
            None,
            **lk_params,
        )

        if pts_1_back is None or status_bwd is None:
            return None, None, None

        p1 = pts_1.reshape(-1, 2)
        p2 = pts_2.reshape(-1, 2)
        p1_back = pts_1_back.reshape(-1, 2)

        status_fwd = status_fwd.reshape(-1).astype(bool)
        status_bwd = status_bwd.reshape(-1).astype(bool)

        err_fwd = err_fwd.reshape(-1) if err_fwd is not None else np.zeros(len(p1))
        fb_error = np.linalg.norm(p1 - p1_back, axis=1)

        valid = (
            status_fwd &
            status_bwd &
            np.isfinite(p2).all(axis=1) &
            (err_fwd < self.lk_max_error) &
            (fb_error < self.fb_max_error)
        )

        return p1[valid], p2[valid], valid

    def _depth_from_sgbm(self, img_left_1: np.ndarray, img_right_1: np.ndarray):
        """
        Строит disparity/depth map через StereoSGBM.
        """
        disp_raw = self.stereo.compute(img_left_1, img_right_1)
        disparity = disp_raw.astype(np.float32) / 16.0

        depth = np.full_like(disparity, np.nan, dtype=np.float32)

        valid = disparity > 1.0

        depth[valid] = self.triangulation.fx * self.baseline / disparity[valid]

        valid_depth = (
            valid &
            np.isfinite(depth) &
            (depth > self.min_depth) &
            (depth < self.max_depth)
        )

        return depth, disparity, valid_depth

    def _build_3d_2d_correspondences(self, pts_1: np.ndarray, pts_2: np.ndarray, depth: np.ndarray):
        """
        Собирает пары 3D_t и 2D_t+1.
        """
        object_points = []
        image_points = []

        H, W = depth.shape

        for p1, p2 in zip(pts_1, pts_2):
            x, y = p1

            # Координаты пикселя для взятия глубины из depth map.
            x_i = int(round(x))
            y_i = int(round(y))

            # Проверяем границы изображения.
            if x_i < 0 or x_i >= W or y_i < 0 or y_i >= H:
                continue

            Z = depth[y_i, x_i]

            # Проверяем глубину.
            if not np.isfinite(Z) or Z <= self.min_depth or Z >= self.max_depth:
                continue

            # Обратная проекция пикселя в 3D camera coordinates.
            X = (x - self.triangulation.cx) * Z / self.triangulation.fx
            Y = (y - self.triangulation.cy) * Z / self.triangulation.fy

            object_points.append([X, Y, Z])
            image_points.append([p2[0], p2[1]])

        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)

        return object_points, image_points

    def _return_torch_result(self, rvec, tvec, debug):
        """
        Конвертирует OpenCV PnP результат в Lie-вектор проекта.
        """
        rvec_pred = torch.from_numpy(rvec.squeeze()).to(dtype=torch.float64)
        tvec_pred = torch.from_numpy(tvec.squeeze()).to(dtype=torch.float64)

        R_pred = RT.from_rotvec(rvec_pred)
        pose_pred = PT.from_rt(R_pred, tvec_pred).inv()

        C = torch.tensor([
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ], dtype=pose_pred.dtype, device=pose_pred.device)
        pose_pred = pose_pred.change_basis(C)
        
        
        print("PnP raw tvec:", tvec.squeeze())
        print("PnP after inv t:", pose_pred.t)
        
        y_pred = pose_pred.as_lie()

        debug["success"] = True
        debug["reason"] = "ok"

        return y_pred, debug

    def forward(self, img_left_1: np.ndarray, img_right_1: np.ndarray, img_left_2: np.ndarray):
        """
        Основной forward.
        """
        with torch.no_grad():
            debug = {
                "success": False,
                "reason": None,
                "num_detected_points": 0,
                "num_lk_tracks": 0,
                "num_depth_valid_pixels": 0,
                "num_pnp_points": 0,
                "num_pnp_inliers": 0,
            }

            # 1. Считаем dense depth map из stereo-пары t.
            depth, disparity, valid_depth = self._depth_from_sgbm(img_left_1, img_right_1)
            debug["num_depth_valid_pixels"] = int(valid_depth.sum())

            # 2. Ищем точки на left_t для LK tracking.
            pts_1_raw = self._detect_points(img_left_1)

            if pts_1_raw is None or len(pts_1_raw) == 0:
                return self._zero_prediction("no_points_for_lk", debug)

            debug["num_detected_points"] = int(len(pts_1_raw))

            # 3. Отслеживаем точки left_t -> left_t+1.
            pts_1, pts_2, valid_lk = self._track_lk(img_left_1, img_left_2, pts_1_raw)

            if pts_1 is None or pts_2 is None:
                return self._zero_prediction("lk_tracking_failed", debug)

            debug["num_lk_tracks"] = int(len(pts_1))

            if len(pts_1) < self.min_pnp_points:
                return self._zero_prediction("not_enough_lk_tracks", debug)

            # 4. Собираем 3D_t -> 2D_t+1 соответствия.
            object_points, image_points = self._build_3d_2d_correspondences(
                pts_1,
                pts_2,
                depth,
            )

            debug["num_pnp_points"] = int(len(object_points))

            if len(object_points) < self.min_pnp_points:
                return self._zero_prediction("not_enough_3d_2d_points", debug)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.triangulation.K,
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            
            if inliers is not None:
                debug["num_pnp_inliers"] = int(len(inliers))

            if not success:
                return self._zero_prediction("pnp_success_false", debug)

            if rvec is None or tvec is None:
                return self._zero_prediction("pnp_empty_pose", debug)

            if inliers is None or len(inliers) < self.min_pnp_inliers:
                return self._zero_prediction("not_enough_pnp_inliers", debug)
            
            y_pred, debug = self._return_torch_result(rvec, tvec, debug)

            if self.return_debug:
                return y_pred, debug

            return y_pred
