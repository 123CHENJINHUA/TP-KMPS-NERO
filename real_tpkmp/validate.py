"""Validate a frozen model on independent demonstrations without refitting."""
from .bootstrap import new_directory, write_json, digest
import argparse
import numpy as np
from .data import read_record
from .learning import load_demos, rebuild, report_rows, plot_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    meta, data = read_record(args.model, 'model')
    rows, mode = load_demos(args.dataset, meta['config'], meta['samples'])
    known = {s['sha256'] for s in meta['training_sources'] + meta['holdout_sources']}
    if mode != meta['gripper_mode'] or any(row['sha256'] in known for row in rows):
        raise ValueError('独立验证需要相同夹爪模式、且未用于训练/原留出的新示教数据')
    output = new_directory(args.out)
    reports, predictions = report_rows(rebuild(meta, data), rows, meta['config'], meta['samples'])
    write_json(output/'validation.json', dict(model_sha256=digest(args.model), independent=reports,
              passed=all(r['passed'] for r in reports), physical_task_success='not_tested'))
    for i, (row, prediction, report) in enumerate(zip(rows, predictions, reports)):
        plot_poses(output/f'independent_{i+1:02d}.png', row['tcp'], prediction, 'Independent demonstration')
        print(f'{i+1}: {report["position_rms_m"]*1000:.2f} mm / '
              f'{np.rad2deg(report["rotation_rms_rad"]):.2f} deg；通过={report["passed"]}')
    print(output/'validation.json')


if __name__ == '__main__':
    main()
