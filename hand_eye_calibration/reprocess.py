"""Recompute saved ChArUco observations with validated RGB intrinsics; no hardware."""
from bootstrap import inside_folder, read_json, write_json, utc_now
import argparse
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import cv2
import numpy as np
from calibration import load_dataset
from vision import camera_matrix, BoardDetector
from motion_checks import check_capture


def reprocess_dataset(dataset, intrinsics, output, max_reprojection_px=1.):
    source = Path(dataset).resolve()
    output = inside_folder(output)
    if output.exists():
        raise ValueError('输出目录已存在；请使用新目录，原数据不会覆盖')
    metadata, samples, digest = load_dataset(source)
    if metadata['mode'] not in ('eye-in-hand', 'eye-to-hand'):
        raise ValueError('只重算已有手眼数据，不接受 intrinsics 数据集')
    intrinsic_path = Path(intrinsics).resolve()
    camera = read_json(intrinsic_path)
    if camera.get('passed') is not True:
        raise ValueError('需要已通过检查的内参（passed=true）')
    raw = metadata['camera_raw']
    keys = ('serial', 'width', 'height', 'fps', 'frame')
    if any(camera.get(k) != raw.get(k) for k in keys):
        raise ValueError('内参的 serial、width、height、fps、frame 必须与原始 RGB 图像一致')
    camera_matrix(camera)
    if not np.isfinite(max_reprojection_px) or max_reprojection_px <= 0:
        raise ValueError('重投影阈值必须为正')
    bc = metadata['board']
    board = cv2.aruco.CharucoBoard(
        (bc['squares_x'], bc['squares_y']), bc['square_length_m'], bc['marker_length_m'],
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])),
        np.array(bc['marker_ids'], np.int32))
    board.setLegacyPattern(bc.get('legacy_pattern', False))
    detector = BoardDetector(board, camera, min_corners=4, max_rms_px=max_reprojection_px)
    updated_metadata = copy.deepcopy(metadata)
    updated_metadata['camera'] = camera
    updated_metadata['reprocessing'] = {
        'created_utc': utc_now(), 'source_dataset': str(source), 'source_dataset_sha256': digest,
        'intrinsics_file': str(intrinsic_path), 'original_camera': metadata['camera'],
        'method': 'redetect all saved RGB images with new intrinsics; preserve original corner ID set',
        'preserved': 'sample IDs, raw RGB images, robot poses, joint angles, capture times and frame metadata',
        'note': 'Derived copy of the same acquisition; not a new independent validation set.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.reprocess_', dir=output.parent))
    accepted = []
    try:
        write_json(staging / 'metadata.json', updated_metadata)
        for sample in samples:
            image_path = (source / sample['image']).resolve()
            if not image_path.is_relative_to(source):
                raise ValueError(f'样本 {sample["id"]} 的图像不在原数据集内')
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (raw['height'], raw['width']):
                raise ValueError(f'样本 {sample["id"]} 图像打不开或尺寸不一致')
            try:
                detection = detector.detect(image)
                original_ids = sample['corner_ids']
                if set(detection['corner_ids']) != set(original_ids):
                    raise ValueError('重检测角点编号集合与原图不同；未丢弃样本或角点')
                indices = {value: i for i, value in enumerate(detection['corner_ids'])}
                corners = [detection['corners_px'][indices[value]] for value in original_ids]
                detection['corner_ids'], detection['corners_px'] = list(original_ids), corners
                detection.update(detector.pose_from_corners(original_ids, corners))
                comparisons = check_capture(
                    sample['robot'], detection, accepted,
                    metadata.get('motion_quality', {}).get('maximum_relative_rotation_disagreement_deg', 8.))
            except (ValueError, cv2.error) as exc:
                raise ValueError(f'样本 {sample["id"]} 重算失败：{exc}；整次输出取消，原数据未改动') from exc
            updated = copy.deepcopy(sample)
            updated.update(detection, motion_comparisons=comparisons)
            relative = image_path.relative_to(source)
            updated['image'] = relative.as_posix()
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            write_json(staging / 'samples' / f'{sample["id"]}.json', updated)
            accepted.append(updated)
        if load_dataset(source)[2] != digest:
            raise ValueError('重算期间原数据发生变化；取消输出')
        if output.exists():
            raise ValueError('输出目录已被创建，取消以免覆盖')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return len(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--intrinsics', required=True)
    parser.add_argument('--out', required=True, help='新目录，限于 hand_eye_calibration 内；不覆盖')
    parser.add_argument('--max-reprojection-px', type=float, default=1.)
    args = parser.parse_args()
    count = reprocess_dataset(args.dataset, args.intrinsics, args.out, args.max_reprojection_px)
    print(f'重算 {count} 个样本到 {Path(args.out).resolve()}；原始图像和机械臂位姿保留，原数据未修改。')
    print('接下来对新目录运行 solve.py；同一批照片的派生副本不算独立验证。')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        sys.exit(f'重算失败：{exc}')
