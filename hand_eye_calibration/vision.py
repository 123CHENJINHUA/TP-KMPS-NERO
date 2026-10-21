"""ChArUco geometry, corner quality and planar PnP for the COLOR optical frame."""
from bootstrap import HERE, read_json
import cv2
import numpy as np
from transforms import checked


def load_board(path=HERE / 'board_config.json', dictionary=None):
    config = read_json(path)
    if dictionary:
        config['dictionary'] = dictionary
    name = config.get('dictionary')
    if not name or not name.startswith('DICT_') or not hasattr(cv2.aruco, name):
        raise ValueError('必须指定实际板的完整字典名：如 DICT_6X6_50 或 DICT_5X5_100；未知时先运行 inspect_board.py')
    sx, sy = int(config['squares_x']), int(config['squares_y'])
    square, marker = float(config['square_length_m']), float(config['marker_length_m'])
    if min(sx, sy) < 3 or not 0 < marker < square:
        raise ValueError('板尺寸无效：至少 3×3，且 0 < marker < square（m）')
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    ids = config.get('marker_ids')
    args = ((sx, sy), square, marker, aruco_dict)
    board = cv2.aruco.CharucoBoard(*args) if ids is None else cv2.aruco.CharucoBoard(*args, np.array(ids, np.int32))
    board.setLegacyPattern(bool(config.get('legacy_pattern', False)))
    numbers = board.getIds().ravel()
    capacity = len(aruco_dict.bytesList)
    if len(set(numbers.tolist())) != len(numbers) or np.any(numbers < 0) or np.any(numbers >= capacity):
        raise ValueError(f'此板有 {len(numbers)} 个 marker，编号范围 {int(numbers.min())}–{int(numbers.max())}；'
                         f'{name} 只有编号 0–{capacity - 1}，或板编号存在重复。请使用与实物一致且容量足够的字典。')
    config['marker_ids'] = numbers.tolist()
    return board, config


def camera_matrix(model):
    K = np.array(model['K'], dtype=float)
    dist = np.array(model['distortion_coefficients'], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('无效的相机内参 K')
    if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError('无效的 OpenCV 畸变参数')
    if model.get('distortion_model') not in ('opencv_brown', 'none'):
        raise ValueError('必须先转换为 OpenCV Brown 模型或使用重新标定的内参')
    return K, dist


class BoardDetector:
    def __init__(self, board, camera=None, min_corners=10, max_rms_px=1.0):
        self.board = board
        self.min_corners = min_corners
        self.max_rms_px = max_rms_px
        cp = cv2.aruco.CharucoParameters()
        self.K = self.dist = None
        if camera is not None:
            self.K, self.dist = camera_matrix(camera)
            cp.cameraMatrix, cp.distCoeffs = self.K, self.dist
        dp = cv2.aruco.DetectorParameters()
        # ChArUco interpolation refines chess corners; refining marker corners
        # independently can bias them towards nearby chess edges.
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.detector = cv2.aruco.CharucoDetector(board, cp, dp)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, marker_ids = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            raise ValueError(f'ChArUco 角点不足：{0 if ids is None else len(ids)}/{self.min_corners}')
        ids = ids.reshape(-1).astype(np.int32)
        uv = corners.reshape(-1, 2).astype(np.float64)
        obj = self.board.getChessboardCorners()[ids].astype(np.float64)
        if len(set(ids.tolist())) != len(ids) or np.linalg.matrix_rank(obj[:, :2] - obj[0, :2]) != 2:
            raise ValueError('角点重复或共线')
        area = float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32))))
        if area < 1500:
            raise ValueError('板在图像中太小；请靠近或提高分辨率')
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 255)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F)[mask > 0].var())
        if blur < 35:
            raise ValueError(f'板图像可能模糊：Laplacian variance={blur:.1f}')
        result = {'corner_ids': ids.tolist(), 'corners_px': uv.tolist(),
                  'marker_ids': [] if marker_ids is None else marker_ids.ravel().tolist(),
                  'hull_area_px2': area, 'sharpness': blur}
        if self.K is None:
            return result
        result.update(self.pose_from_corners(ids, uv))
        return result

    def pose_from_corners(self, ids, uv):
        obj = self.board.getChessboardCorners()[np.asarray(ids, np.int32)].astype(np.float64)
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if len(obj) < self.min_corners or np.linalg.matrix_rank(obj[:, :2] - obj[0, :2]) != 2:
            raise ValueError('PnP 至少需要足够数量的非共线角点')
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, uv, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        if ok:
            for rvec, tvec in zip(rvecs, tvecs):
                R = cv2.Rodrigues(rvec)[0]
                if np.min((obj @ R.T + tvec.ravel())[:, 2]) <= 0:
                    continue
                projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
                rms = float(np.sqrt(np.mean(np.sum((projected - uv) ** 2, axis=1))))
                candidates.append((rms, rvec, tvec))
        if not candidates:
            raise ValueError('PnP 未找到位于相机前方的有效解')
        candidates.sort(key=lambda c: c[0])
        best, rvec, tvec = candidates[0]
        if len(candidates) > 1:
            second, other_r, _ = candidates[1]
            separation = np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(rvec)[0].T @ cv2.Rodrigues(other_r)[0])[0])
            if second - best < 0.08 and separation > np.deg2rad(5):
                raise ValueError('平面 PnP 两个解过于接近；请改变倾斜角度后采样')
        rvec, tvec = cv2.solvePnPRefineLM(obj, uv, self.K, self.dist, rvec, tvec)
        projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - uv) ** 2, axis=1))))
        if not np.isfinite(rms) or rms > self.max_rms_px:
            raise ValueError(f'重投影误差 {rms:.3f}px > {self.max_rms_px}px')
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return {'T_color_board': checked(T).tolist(), 'reprojection_rms_px': rms}

    def draw(self, image, detection):
        result = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(result, np.array(detection['corners_px'], np.float32).reshape(-1, 1, 2),
                                             np.array(detection['corner_ids'], np.int32).reshape(-1, 1))
        if 'T_color_board' in detection:
            T = np.array(detection['T_color_board'])
            cv2.drawFrameAxes(result, self.K, self.dist, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], .06)
        return result
