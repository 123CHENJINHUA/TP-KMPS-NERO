"""Meter-scale STL geometry and independent CPU z-buffer checks."""
import re
import struct
import itertools
from pathlib import Path
import numpy as np
from ..bootstrap import digest
from .common import checked


def load_mesh(cfg):
    path = Path(cfg['mesh_path'])
    raw = path.read_bytes()
    if cfg.get('mesh_sha256') and digest(path) != cfg['mesh_sha256']:
        raise ValueError('CAD 校验值不一致；更换模型需要更新配置并重新绑定抓取关系')
    count = struct.unpack('<I', raw[80:84])[0] if len(raw) >= 84 else -1
    if count > 0 and len(raw) == 84+50*count:
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].astype(float)
    else:
        values = re.findall(rb'\bvertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)', raw)
        triangles = np.array(values, float).reshape(-1, 3, 3)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError('需要有效的 STL 三角网格')
    triangles *= .001 if cfg['mesh_units'] == 'mm' else 1.
    extents = np.ptp(triangles.reshape(-1, 3), axis=0)
    if not np.allclose(extents, cfg['expected_extents_m'], rtol=.02, atol=.0001):
        raise ValueError(f'CAD 实际尺寸 {extents.tolist()}m 与配置不符，检查毫米/米')
    return triangles


def bounds(triangles):
    points = triangles.reshape(-1, 3)
    return np.array([points.min(axis=0), points.max(axis=0)])


def corners(box):
    return np.array(list(itertools.product(*zip(box[0], box[1]))))


def render_depth(triangles, pose, K, shape):
    """Perspective-correct triangle rasterization; quality check, not learned pose."""
    pose = checked(pose)
    height, width = shape
    image = np.full((height, width), np.inf)
    points = triangles@pose[:3, :3].T+pose[:3, 3]
    for triangle in points:
        if np.min(triangle[:, 2]) <= .001:
            continue
        projected = triangle@np.asarray(K).T
        uv = projected[:, :2]/projected[:, 2:]
        low = np.maximum(np.floor(uv.min(axis=0)).astype(int), [0, 0])
        high = np.minimum(np.ceil(uv.max(axis=0)).astype(int), [width-1, height-1])
        if np.any(low > high):
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
        (x0, y0), (x1, y1), (x2, y2) = uv
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denominator) < 1e-10:
            continue
        a = ((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/denominator
        b = ((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/denominator
        c = 1-a-b
        valid = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        reciprocal = a/triangle[0, 2]+b/triangle[1, 2]+c/triangle[2, 2]
        depth = np.divide(1., reciprocal, out=np.full_like(reciprocal, np.inf), where=valid & (reciprocal > 0))
        patch = image[low[1]:high[1]+1, low[0]:high[0]+1]
        np.minimum(patch, depth, out=patch)
    image[~np.isfinite(image)] = 0
    return image


def quality(triangles, pose, depth, mask, K, cfg):
    import cv2
    height, width = depth.shape
    scale = min(1., 400/width)
    size = (int(round(width*scale)), int(round(height*scale)))
    low_depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    low_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    reduced_K = np.diag([size[0]/width, size[1]/height, 1.])@K
    rendered = render_depth(triangles, pose, reduced_K, low_depth.shape)
    silhouette = rendered > 0
    valid = (low_depth >= cfg['min_depth_m']) & (low_depth <= cfg['max_depth_m'])
    overlap = silhouette & low_mask & valid
    union = np.count_nonzero(silhouette | low_mask)
    iou = np.count_nonzero(silhouette & low_mask)/max(1, union)
    mask_valid = np.count_nonzero(valid & low_mask)/max(1, np.count_nonzero(low_mask))
    differences = abs(rendered[overlap]-low_depth[overlap])
    median = float(np.median(differences)) if len(differences) else None
    p90 = float(np.percentile(differences, 90)) if len(differences) else None
    reasons = []
    if np.count_nonzero(mask) < cfg['min_mask_pixels']:
        reasons.append('mask_too_small')
    if iou < cfg['min_render_iou']:
        reasons.append('render_mask_iou')
    if mask_valid < cfg['min_mask_depth_fraction']:
        reasons.append('insufficient_valid_depth')
    required = max(20, int(cfg['min_depth_overlap_pixels']*scale**2))
    if len(differences) < required:
        reasons.append('insufficient_depth_overlap')
    if median is None or median > cfg['max_depth_median_error_m']:
        reasons.append('depth_median_error')
    if p90 is None or p90 > cfg['max_depth_p90_error_m']:
        reasons.append('depth_p90_error')
    return dict(passed=not reasons, reasons=reasons, render_mask_iou=float(iou),
                mask_valid_depth_fraction=float(mask_valid), overlap_pixels=len(differences),
                depth_median_error_m=median, depth_p90_error_m=p90)
