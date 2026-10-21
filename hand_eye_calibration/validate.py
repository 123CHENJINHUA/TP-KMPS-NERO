"""Validate saved transforms, without re-estimating their constant companion frame."""
from bootstrap import read_json, write_json, utc_now
import argparse
import sys
import numpy as np
import cv2
from calibration import load_dataset, signature, evaluate, passed, print_report, distinct_count
from transforms import checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--dataset', help='独立验证数据；不提供时检查原数据的留出样本')
    p.add_argument('--out', help='可选：另存验证报告 JSON')
    args = p.parse_args()
    result = read_json(args.result)
    metadata, samples, digest = load_dataset(args.dataset or result['dataset'])
    if signature(metadata) != result['signature']:
        raise ValueError('模式、相机/内参、板、机械臂身份/位姿来源不一致；不能混用')
    if args.dataset is None:
        if digest != result['dataset_sha256']:
            raise ValueError('原数据在求解后发生变化；请使用未修改的数据或明确指定独立验证集')
        samples = [s for s in samples if s['id'] in result['holdout_ids']]
    elif digest == result['dataset_sha256']:
        raise ValueError('--dataset 应指定独立采集的数据，原数据用默认的留出验证')
    if distinct_count([checked(s['robot']['T_base_flange']) for s in samples]) < 5:
        raise ValueError('验证至少需要 5 个不同位姿')
    bc = metadata['board']
    board = cv2.aruco.CharucoBoard((bc['squares_x'], bc['squares_y']), bc['square_length_m'], bc['marker_length_m'],
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])), np.array(bc['marker_ids'], np.int32))
    board.setLegacyPattern(bc.get('legacy_pattern', False))
    X = checked(result[result['transform_name']]['matrix'])
    F = checked(result[result['constant_transform_name']]['matrix'])
    report = evaluate(samples, result['mode'], X, F, board, metadata['camera'])
    good = passed(report, result['limits'])
    print_report('独立验证' if args.dataset else '留出验证', report)
    print(f'阈值通过: {good}；未重新估计外参或板偏置')
    if args.out:
        write_json(args.out, {'created_utc': utc_now(), 'passed': good, 'report': report, 'limits': result['limits']})
    return 0 if good else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        sys.exit(f'验证失败：{exc}')
