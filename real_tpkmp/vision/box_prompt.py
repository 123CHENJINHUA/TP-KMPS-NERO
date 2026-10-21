"""Box geometry shared with isolated SAM3 workers; no robot or model imports."""
import numpy as np


def checked_box(box):
    box = np.asarray(box, float)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('框需要归一化 XYXY：[x0,y0,x1,y1]，范围 0–1 且宽高为正')
    return box


def mask_box(mask, margin=.08):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('分割掩膜为空')
    h, w = mask.shape
    bounds = np.array([x.min()/w, y.min()/h, (x.max()+1)/w, (y.max()+1)/h])
    padding = np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]])*margin+np.array([2/w, 2/h])
    return checked_box(np.clip(bounds+np.r_[-padding, padding], 0, 1))


def select_box_candidate(masks, scores, box, threshold=.5):
    """Retain the in-box instance; reject outside or ambiguous masks."""
    box = checked_box(box)
    masks, scores = np.asarray(masks, bool), np.asarray(scores, float).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError('SAM3 mask/score 格式不一致')
    candidates = []
    for i in np.flatnonzero(np.isfinite(scores) & (scores >= threshold)):
        if not masks[i].any():
            continue
        candidate = mask_box(masks[i], margin=0.)
        lo, hi = np.maximum(box[:2], candidate[:2]), np.minimum(box[2:], candidate[2:])
        overlap = float(np.prod(np.maximum(0., hi-lo)))
        area = float(np.prod(candidate[2:]-candidate[:2]))
        union = float(np.prod(box[2:]-box[:2]))+area-overlap
        iou = overlap/max(union, 1e-12)
        center = (candidate[:2]+candidate[2:])/2
        if overlap/max(area, 1e-12) >= .5 and iou >= .15 and np.all(center >= box[:2]) and np.all(center <= box[2:]):
            candidates.append((iou, float(scores[i]), int(i)))
    candidates.sort(reverse=True)
    if not candidates:
        raise ValueError('框内没有可靠目标；请把框贴近完整物体后重新框选')
    if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .05:
        first, second = masks[candidates[0][2]], masks[candidates[1][2]]
        duplicate_iou = np.count_nonzero(first & second)/max(1, np.count_nonzero(first | second))
        if duplicate_iou < .9:
            raise ValueError('框内有多个相近候选，请缩小框，只包含一个物体')
    return candidates[0][2]


def select_box(rgb):
    """Select on an exact frozen RGB frame, returning normalized XYXY or None."""
    import cv2
    import os
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise ValueError('框选需要桌面窗口；无窗口模式请提供 --box x0 y0 x1 y1（归一化坐标）')
    # Limit the dialog's image itself; no ambiguity from window scaling coordinates.
    h, w = rgb.shape[:2]
    scale = min(1., 960/w, 640/h)
    image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(round(w*scale)), int(round(h*scale))))
    name = 'Select ONE object - drag, Enter confirm, C cancel'
    try:
        rect = cv2.selectROI(name, image, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(name)
    x, y, width, height = rect
    if width < 4 or height < 4:
        return None
    ih, iw = image.shape[:2]
    return checked_box([x/iw, y/ih, (x+width)/iw, (y+height)/ih]).tolist()


class MovingBox:
    """Update prompts using target-local optical flow. A lost target needs reselection."""
    def __init__(self, box=None):
        self.box = checked_box(box).copy() if box is not None else None
        self.gray, self.mask = None, None

    def clear(self):
        self.box, self.gray, self.mask = None, None, None

    def select(self, box):
        self.box = checked_box(box).copy()
        self.gray, self.mask = None, None

    def predict(self, rgb):
        import cv2
        if self.box is None:
            raise ValueError('需要按 F 重新框选目标')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        box = self.box.copy()
        if self.gray is None or self.gray.shape != gray.shape:
            return box
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=100, qualityLevel=.01,
                                        minDistance=3, mask=self.mask.astype(np.uint8)*255)
        if points is None or len(points) < 4:
            return box
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None,
                                                    winSize=(31, 31), maxLevel=3)
        if following is None:
            return box
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, following, None,
                                                        winSize=(31, 31), maxLevel=3)
        if backward is None:
            return box
        good = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0)
        good &= np.linalg.norm(backward-points, axis=2).reshape(-1) < 2.
        if good.sum() < 4:
            return box
        affine, inliers = cv2.estimateAffinePartial2D(points[good], following[good], method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.)
        if affine is None or inliers is None or inliers.sum()/len(inliers) < .5:
            return box
        factor = np.sqrt(abs(np.linalg.det(affine[:, :2])))
        if not .6 < factor < 1.6:
            return box
        h, w = gray.shape
        corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])*[w, h]
        mapped = corners@affine[:, :2].T+affine[:, 2]
        updated = np.r_[mapped.min(axis=0)/[w, h], mapped.max(axis=0)/[w, h]]
        updated = np.clip(updated, 0., 1.)
        return checked_box(updated)

    def accept(self, rgb, mask, box=None):
        import cv2
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.mask = np.asarray(mask, bool).copy()
        self.box = checked_box(box).copy() if box is not None else mask_box(self.mask)
