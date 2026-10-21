from ..bootstrap import HERE, inside, read_json, digest, write_json, new_directory, utc_now
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = HERE / 'vision'


def output_path(path):
    value = inside(path)
    if not value.is_relative_to(ROOT):
        raise ValueError(f'视觉输出必须位于 {ROOT}')
    return value


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def object_config(path=None):
    filename = Path(path).resolve() if path else ROOT/'objects/keel.json'
    cfg = read_json(filename)
    cfg['mesh_path'] = str(resolve_path(cfg['mesh_path']))
    if cfg.get('mesh_units') not in ('mm', 'm'):
        raise ValueError('CAD 必须明确 mesh_units=mm 或 m')
    if not cfg.get('object_id') or cfg.get('schema_version') != 1:
        raise ValueError('物体配置需要 schema_version=1 和 object_id')
    expected = np.asarray(cfg['expected_extents_m'], float)
    if expected.shape != (3,) or not np.isfinite(expected).all() or np.any(expected <= 0):
        raise ValueError('必须配置已确认的三轴 CAD 尺寸 expected_extents_m')
    return cfg


def checked(matrix):
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError('需要有限的右手刚体变换 4x4；拒绝缩放、反射或未归一化旋转')
    return matrix


def camera_model(model):
    K = np.asarray(model['K'], float)
    dist = np.asarray(model.get('distortion_coefficients', [0.]*5), float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.isfinite(dist).all()
            or dist.shape not in ((4,), (5,), (8,), (12,), (14,))):
        raise ValueError('相机内参或畸变系数无效')
    if model.get('frame') != 'color_optical':
        raise ValueError('需要 color_optical 相机坐标系')
    if model.get('distortion_model') not in ('none', 'opencv_brown'):
        raise ValueError('只支持已标定的 OpenCV Brown 或零畸变内参')
    if model.get('distortion_model') == 'none' and np.any(dist):
        raise ValueError('none 畸变模型不能包含非零系数')
    return K, dist


def pose_average(matrices):
    matrices = np.array([checked(m) for m in matrices])
    result = np.eye(4)
    result[:3, 3] = np.mean(matrices[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(matrices[:, :3, :3]).mean().as_matrix()
    errors = [(np.linalg.norm(m[:3, 3]-result[:3, 3]),
               Rotation.from_matrix(result[:3, :3].T@m[:3, :3]).magnitude()) for m in matrices]
    return result, np.asarray(errors)


def read_capture(path):
    directory = Path(path).resolve()
    meta = read_json(directory/'capture.json')
    if meta.get('schema_version') != 1 or meta.get('kind') != 'rgbd_capture' or not isinstance(meta.get('synthetic'), bool):
        raise ValueError('不是有效的 RGB-D 采集目录')
    if digest(directory/'rgbd.npz') != meta['rgbd_sha256']:
        raise ValueError('RGB-D 数据在采集后被修改')
    with np.load(directory/'rgbd.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    rgb, depth, K = data['rgb'], data['depth_m'], data['K']
    n, h, w, c = rgb.shape
    if (c != 3 or rgb.dtype != np.uint8 or depth.shape != (n, h, w) or K.shape != (3, 3)
            or not np.isfinite(depth).all() or np.any(depth < 0) or len(meta['frames']) != n):
        raise ValueError('RGB/深度分辨率、格式或帧数不一致')
    if not np.allclose(K, meta['camera']['K']):
        raise ValueError('图像与相机矩阵不匹配')
    camera_model(meta['camera'])
    if (meta['camera'].get('rectified') is not True or meta['camera'].get('registered_depth') is not True
            or meta['camera'].get('distortion_model') != 'none' or np.any(meta['camera']['distortion_coefficients'])):
        raise ValueError('网络输入必须是已去畸变的 COLOR 图像和配准到相同像素的米制深度')
    return directory, meta, data
