"""Hand-eye solution with a held-out set that is excluded from all fitting."""
from bootstrap import write_json, utc_now
import argparse
from pathlib import Path
import sys
import numpy as np
from calibration import load_dataset, signature, fit, evaluate, passed, print_report, METHODS
from transforms import describe, inverse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', help='默认 <dataset>/result.json；不覆盖已有文件')
    p.add_argument('--method', choices=METHODS, default='PARK')
    p.add_argument('--no-refine', action='store_true')
    p.add_argument('--holdout-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-translation-mm', type=float, default=5.)
    p.add_argument('--max-rotation-deg', type=float, default=2.)
    p.add_argument('--max-loop-reprojection-px', type=float, default=5.)
    args = p.parse_args()
    if not 0 < args.holdout_fraction < .5 or min(args.max_translation_mm, args.max_rotation_deg, args.max_loop_reprojection_px) <= 0:
        p.error('留出比例必须在 (0,0.5)，误差阈值必须为正')
    metadata, samples, digest = load_dataset(args.dataset)
    mode = metadata['mode']
    if mode == 'intrinsics':
        raise ValueError('内参数据请使用 calibrate_intrinsics.py')
    ntest = max(5, int(round(len(samples) * args.holdout_fraction)))
    if len(samples) - ntest < 12:
        raise ValueError('总样本不足：至少 12 张训练 + 5 张留出，建议采集 30–40 张')
    order = np.random.default_rng(args.seed).permutation(len(samples))
    test = [samples[i] for i in sorted(order[:ntest])]
    train = [samples[i] for i in sorted(order[ntest:])]
    X, F, observability, refinement = fit(train, mode, args.method, not args.no_refine)
    # Board configuration is part of dataset metadata, not today's mutable config.
    from vision import cv2
    bc = metadata['board']
    board = cv2.aruco.CharucoBoard((bc['squares_x'], bc['squares_y']), bc['square_length_m'], bc['marker_length_m'],
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])),
                                  np.array(bc['marker_ids'], np.int32))
    board.setLegacyPattern(bc.get('legacy_pattern', False))
    train_report = evaluate(train, mode, X, F, board, metadata['camera'])
    test_report = evaluate(test, mode, X, F, board, metadata['camera'])
    limits = {'translation_rms_mm': args.max_translation_mm, 'rotation_rms_deg': args.max_rotation_deg,
              'loop_reprojection_rms_px': args.max_loop_reprojection_px}
    name = 'T_flange_color' if mode == 'eye-in-hand' else 'T_base_color'
    constant_name = 'T_base_board' if mode == 'eye-in-hand' else 'T_flange_board'
    result = {'schema_version': 1, 'created_utc': utc_now(), 'mode': mode, 'synthetic': metadata['synthetic'],
              'dataset': str(Path(args.dataset).resolve()), 'dataset_sha256': digest, 'signature': signature(metadata),
              'method': args.method, 'refinement': refinement, 'observability': observability,
              'transform_name': name, name: describe(X), 'inverse_transform': describe(inverse(X)),
              'constant_transform_name': constant_name, constant_name: describe(F),
              'train_ids': [s['id'] for s in train], 'holdout_ids': [s['id'] for s in test],
              'train_report': train_report, 'holdout_report': test_report, 'limits': limits,
              'passed': passed(test_report, limits),
              'convention': 'T_parent_child maps child into parent; translation m; quaternion xyzw; COLOR optical +X right +Y down +Z forward'}
    out = args.out or str(Path(args.dataset) / 'result.json')
    write_json(out, result)
    print_report('训练集', train_report)
    print_report('留出验证集（未参与求解）', test_report)
    print(f'{name} =\n{X}\n保存 {out}\n留出阈值通过: {result["passed"]}')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        sys.exit(f'求解失败：{exc}')
