"""Optional ChArUco RGB intrinsic calibration with held-out reprojection validation."""
from bootstrap import write_json, utc_now
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from calibration import load_dataset
from vision import BoardDetector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='collect.py --mode intrinsics 的目录')
    p.add_argument('--out', help='默认 <dataset>/intrinsics.json')
    p.add_argument('--max-rms-px', type=float, default=1.)
    args = p.parse_args()
    metadata, samples, _ = load_dataset(args.dataset)
    if metadata['mode'] != 'intrinsics' or len(samples) < 20:
        raise ValueError('需至少 20 张 intrinsics 模式的不同位置/角度图片，建议 30–40 张')
    bc = metadata['board']
    board = cv2.aruco.CharucoBoard((bc['squares_x'], bc['squares_y']), bc['square_length_m'], bc['marker_length_m'],
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])), np.array(bc['marker_ids'], np.int32))
    board.setLegacyPattern(bc.get('legacy_pattern', False))
    order = np.random.default_rng(42).permutation(len(samples))
    ntest = max(5, len(samples) // 5)
    train = [samples[i] for i in order[ntest:]]
    test = [samples[i] for i in order[:ntest]]
    objects = [board.getChessboardCorners()[np.array(s['corner_ids'], np.int32)].astype(np.float32) for s in train]
    images = [np.array(s['corners_px'], np.float32) for s in train]
    raw = metadata['camera_raw']
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, images, (raw['width'], raw['height']), None, None)
    camera = {key: raw[key] for key in ('serial', 'width', 'height', 'fps', 'frame')}
    camera.update(K=K.tolist(), distortion_model='opencv_brown', distortion_coefficients=dist.ravel().tolist(),
                  created_utc=utc_now(), source='charuco_rgb_calibration', dataset=str(Path(args.dataset).resolve()),
                  train_rms_px=float(rms), train_ids=[s['id'] for s in train], holdout_ids=[s['id'] for s in test])
    detector = BoardDetector(board, camera, max_rms_px=max(args.max_rms_px * 10, 10))
    errors = [detector.pose_from_corners(s['corner_ids'], s['corners_px'])['reprojection_rms_px'] for s in test]
    holdout = float(np.sqrt(np.mean(np.square(errors))))
    camera.update(holdout_rms_px=holdout, holdout_errors_px=errors,
                  passed=bool(rms <= args.max_rms_px and holdout <= args.max_rms_px))
    out = args.out or str(Path(args.dataset) / 'intrinsics.json')
    write_json(out, camera)
    print(f'RGB 内参训练 RMS={rms:.3f}px，留出 RMS={holdout:.3f}px，阈值通过={camera["passed"]}\n{out}')
    return 0 if camera['passed'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        sys.exit(f'内参标定失败：{exc}')
