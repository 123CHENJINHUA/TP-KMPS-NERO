"""Dependency check; enumerate RealSense; optional receive-only CAN/FK check."""
import bootstrap
import argparse
import importlib
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can', choices=['can0', 'can1'])
    parser.add_argument('--firmware', default='v121')
    parser.add_argument('--joint-source', choices=['measured', 'leader', 'both'], default='both',
                        help='默认 both 仅检查两种来源；采集仍需明确选择实体臂对应的来源')
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--report', help='可选：保存诊断 JSON，限于手眼标定目录')
    parser.add_argument('--camera-profiles', action='store_true', help='枚举 RGB BGR8 支持的分辨率/帧率，不启动图像流')
    args = parser.parse_args()
    if args.report and not args.can:
        parser.error('--report 需要同时指定 --can')
    good = True
    for module in ('numpy', 'scipy', 'cv2', 'pyrealsense2', 'can', 'matplotlib'):
        try:
            value = importlib.import_module(module)
            print(f'{module}: OK {getattr(value, "__version__", "")} {value.__file__}')
        except ImportError as exc:
            print(f'{module}: MISSING {exc}')
            good = False
    if not good:
        return 2
    from realsense_camera import list_devices
    devices = list_devices(include_profiles=args.camera_profiles)
    print('RealSense devices: ' + json.dumps(devices, ensure_ascii=False))
    from robot_pose import load_sdk
    print(f'Project pyAgxArm: {load_sdk().__version__}')
    if args.can:
        from robot_pose import CanPoseReader
        if not 0 < args.seconds <= 60:
            parser.error('--seconds 必须为 (0,60]')
        reader = CanPoseReader(args.can, args.firmware, 'measured' if args.joint_source == 'both' else args.joint_source)
        sources = ('measured', 'leader') if args.joint_source == 'both' else (args.joint_source,)
        report = {s: {'valid_samples': 0, 'latest': None, 'last_error': None} for s in sources}
        try:
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                status = reader.inspect_sources()
                for source in sources:
                    entry = status[source]
                    if entry['available']:
                        report[source]['valid_samples'] += 1
                        report[source]['latest'] = entry['packet']
                        report[source]['last_error'] = None
                    else:
                        report[source]['last_error'] = entry['error']
                time.sleep(.02)
        finally:
            reader.close()
        summary = {'channel': args.can, 'read_only': True, 'sources': report,
                   'sources_with_valid_samples': [s for s in sources if report[s]['valid_samples'] > 0],
                   'note': '来源可读不证明板/相机安装于该实体臂；采集不会自动选择来源'}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.report:
            bootstrap.write_json(args.report, summary)
        if not summary['sources_with_valid_samples']:
            return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        sys.exit(f'检查失败：{exc}')
