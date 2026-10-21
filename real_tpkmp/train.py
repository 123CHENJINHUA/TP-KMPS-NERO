"""Train Cartesian TP-KMP; validate on entire demonstrations excluded from fitting."""
from .bootstrap import new_directory, write_json, utc_now
import argparse
import numpy as np
from .data import config, physical_signature, save_record
from .learning import load_demos, build_model, report_rows, plot_poses
from .kinematics import pose_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config')
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--components', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = config(args.config)
    if not 1 <= args.components <= args.samples:
        parser.error('components 需要在 1..samples 范围内')
    rows, mode = load_demos(args.dataset, cfg, args.samples)
    order = np.random.default_rng(args.seed).permutation(len(rows))
    count = max(1, int(np.ceil(len(rows)*.2)))
    held = [rows[i] for i in order[:count]]
    train = [rows[i] for i in order[count:]]
    demonstrations = np.array([np.c_[row['phase'], row['tcp']] for row in train])
    frames = np.array([row['frames'] for row in train])
    output = new_directory(args.out)
    print(f'训练 {len(train)} 组；整条留出 {len(held)} 组；每条对齐 {args.samples} 点；GMM={args.components}', flush=True)
    model = build_model(demonstrations, frames, args.samples, args.components)
    reports, predictions = report_rows(model, held, cfg, args.samples)
    train_reports, _ = report_rows(model, train, cfg, args.samples)
    passed = all(item['passed'] for item in reports)
    novel = [all(any(pose_distance > .003 or angle > .03 for pose_distance, angle in
                     [pose_error(a, b) for a, b in zip(row['frames'][1:], tr['frames'][1:])])
                 for tr in train) for row in held]
    report = dict(training=train_reports, holdout=reports, passed=passed, holdout_has_new_frames=novel,
                  validation_type='whole_demonstration_holdout', physical_task_success='not_tested',
                  seed=args.seed, created_utc=utc_now())
    write_json(output/'validation.json', report)
    for i, (row, prediction) in enumerate(zip(held, predictions)):
        plot_poses(output/f'holdout_{i+1:02d}.png', row['tcp'], prediction, 'Whole-demo holdout')
        print(f'留出 {i+1}: {reports[i]["position_rms_m"]*1000:.2f} mm / '
              f'{np.rad2deg(reports[i]["rotation_rms_rad"]):.2f} deg；通过={reports[i]["passed"]}')
    meta = dict(schema_version=1, kind='model', robot='nero', synthetic=any(r['meta']['synthetic'] for r in rows),
                created_utc=utc_now(), config=cfg, signature=physical_signature(cfg),
                samples=args.samples, components=args.components, gripper_mode=mode,
                validation_passed=passed, training_sources=[dict(path=r['path'], sha256=r['sha256']) for r in train],
                holdout_sources=[dict(path=r['path'], sha256=r['sha256']) for r in held], task='pick_place')
    save_record(output/'model.npz', meta, demonstrations=demonstrations, frames=frames,
                reference_joints=np.array([r['joints'] for r in train]),
                gripper_commands=np.array([r['command'] for r in train]),
                gripper_forces_N=np.array([r['force'] for r in train]),
                stage_durations_s=np.array([r['durations'] for r in train]))
    print(f'模型：{output / "model.npz"}\n留出阈值通过：{passed}；轨迹拟合通过不代表抓取成功。')


if __name__ == '__main__':
    main()
