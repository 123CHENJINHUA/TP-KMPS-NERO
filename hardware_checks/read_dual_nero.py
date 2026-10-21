"""Read two Nero arms and AGX teaching grippers. Sends no motion/mode commands."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time

from probe_can import ROOT, inside_project

VENDOR = Path(__file__).resolve().parent / '.vendor'
SDK_ROOT = ROOT / 'pyAgxArm'
JOINT_PARTS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')
LEADER_PARTS_NEW = ('leader_joint_12', 'leader_joint_34', 'leader_joint_56', 'leader_joint_7')
LEADER_PARTS_OLD = tuple(f'leader_joint_{number}' for number in range(1, 8))


def load_sdk():
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(SDK_ROOT))
    try:
        importlib.import_module('can')
    except ImportError as exc:
        raise RuntimeError('python-can missing; run bash hardware_checks/install_deps.sh') from exc
    try:
        sdk = importlib.import_module('pyAgxArm')
    except ImportError as exc:
        raise RuntimeError(f'Could not import project pyAgxArm: {exc}') from exc
    return sdk


def complete_parts(parser, names):
    return all(getattr(parser, name, None) is not None for name in names)


def read_joint_feedback(robot, getter, parts):
    # This project's SDK assembles a 7-value list even when only one CAN part
    # has arrived. Requiring all four/seven parser parts avoids fake startup zeros.
    if not complete_parts(robot._parser, parts):
        return None
    message = getattr(robot, getter)()
    if message is None:
        return None
    angles = [float(value) for value in list(message.msg)]
    if len(angles) != 7 or not all(math.isfinite(value) for value in angles):
        return None
    return {'rad': angles, 'sdk_timestamp': float(message.timestamp),
            'sdk_hz': float(message.hz)}


def read_gripper_feedback(effector):
    message = effector.get_gripper_status()
    if message is None:
        return None
    state = message.msg
    value = float(state.value)
    force = float(state.force)
    if not math.isfinite(value) or not math.isfinite(force):
        return None
    return {'value': value, 'mode': str(state.mode), 'force_N': force,
            'sdk_timestamp': float(message.timestamp), 'sdk_hz': float(message.hz)}


def read_leader_gripper_feedback(effector):
    # The teaching gripper broadcasts a 0x159 command echo. This is distinct
    # from the physical gripper's 0x2A8 measured status.
    message = effector.get_gripper_ctrl_states()
    if message is None:
        return None
    state = message.msg
    status_code = int(state.status_code)
    if status_code not in range(8):
        return None
    mode = 'angle' if status_code & 4 else 'width'
    # The vendored SDK scales 0x159 as metres even in angle mode. Its wire
    # format uses millidegrees in that mode, so correct the decoded value.
    value = float(state.value) * (1000. if mode == 'angle' else 1.)
    force = float(state.force)
    if not math.isfinite(value) or not math.isfinite(force):
        return None
    return {'value': value, 'mode': mode, 'force_N': force,
            'status_code': status_code, 'set_zero': int(state.set_zero),
            'sdk_timestamp': float(message.timestamp), 'sdk_hz': float(message.hz)}


def validate_channels(channels):
    for channel in channels:
        flags_path = Path('/sys/class/net') / channel / 'flags'
        try:
            flags = int(flags_path.read_text().strip(), 16)
        except OSError as exc:
            raise RuntimeError(f'{channel} not found; connect and configure both CAN adapters') from exc
        if not flags & 1:
            raise RuntimeError(f'{channel} is DOWN; bring it up before this read-only test')


def run(args):
    sdk = load_sdk()
    if args.self_check:
        print(f'pyAgxArm {sdk.__version__}; python-can import OK; source={SDK_ROOT}')
        return 0
    validate_channels(args.channels)
    connected = []
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'firmware_profile': args.firmware, 'channels': {}, 'samples': []}
    leader_parts = LEADER_PARTS_OLD if args.firmware in ('default', 'v111') else LEADER_PARTS_NEW
    try:
        for channel in args.channels:
            cfg = sdk.create_agx_arm_config(
                robot=sdk.ArmModel.NERO, firmeware_version=args.firmware,
                interface='socketcan', channel=channel, auto_connect=False,
                enable_check_can=False,
            )
            robot = sdk.AgxArmFactory.create_arm(cfg)
            effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            robot.connect()
            connected.append((channel, robot, effector))
            report['channels'][channel] = {
                'joint_updates': 0, 'leader_updates': 0,
                'gripper_updates': 0, 'leader_gripper_updates': 0,
                'latest': {}, 'last_sdk_timestamps': {},
            }
        print('Receive only: connected to ' + ', '.join(args.channels))
        started = time.monotonic()
        next_print = started
        while time.monotonic() - started < args.seconds:
            sample = {'t_s': round(time.monotonic() - started, 4),
                      'utc': datetime.now(timezone.utc).isoformat(), 'arms': {}}
            for channel, robot, effector in connected:
                measured = read_joint_feedback(robot, 'get_joint_angles', JOINT_PARTS)
                leader = read_joint_feedback(robot, 'get_leader_joint_angles', leader_parts)
                gripper = read_gripper_feedback(effector)
                leader_gripper = read_leader_gripper_feedback(effector)
                arm_sample = {'measured': measured, 'leader': leader,
                              'gripper': gripper, 'leader_gripper': leader_gripper}
                sample['arms'][channel] = arm_sample
                summary = report['channels'][channel]
                for name, data in arm_sample.items():
                    if data is None:
                        continue
                    summary['latest'][name] = data
                    stamp = data['sdk_timestamp']
                    if stamp > 0 and stamp != summary['last_sdk_timestamps'].get(name):
                        summary[name + '_updates' if name != 'measured' else 'joint_updates'] += 1
                        summary['last_sdk_timestamps'][name] = stamp
            report['samples'].append(sample)
            if time.monotonic() >= next_print:
                for channel in args.channels:
                    latest = report['channels'][channel]['latest']
                    measured = latest.get('measured')
                    leader = latest.get('leader')
                    gripper = latest.get('gripper')
                    leader_gripper = latest.get('leader_gripper')
                    def joint_text(data):
                        return ('[' + ', '.join(f'{x:+.3f}' for x in data['rad']) + ']'
                                if data else 'waiting')
                    def grip_text(data):
                        return (f"{data['value']:.4f} {data['mode']}, {data['force_N']:.2f} N"
                                if data else 'waiting')
                    print(f'{channel}: measured(rad)={joint_text(measured)}; '
                          f'leader(rad)={joint_text(leader)}; '
                          f'leader_gripper={grip_text(leader_gripper)}; '
                          f'physical_gripper={grip_text(gripper)}', flush=True)
                next_print = time.monotonic() + 1.
            time.sleep(max(0., 1. / args.rate - (time.monotonic() - started - sample['t_s'])))
    finally:
        for channel, robot, _ in reversed(connected):
            try:
                robot.disconnect()
            except Exception as exc:
                print(f'{channel}: disconnect warning: {exc}', file=sys.stderr)
    for channel, summary in report['channels'].items():
        summary['passed'] = bool(summary['leader_updates']
                                 and summary['leader_gripper_updates'])
        summary.pop('last_sdk_timestamps')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    destination = inside_project(args.report) if args.report else (
        ROOT / 'hardware_checks' / 'reports' /
        ('read_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Report: {destination}')
    for channel, summary in report['channels'].items():
        print(f"{channel}: measured={summary['joint_updates']}, leader={summary['leader_updates']}, "
              f"leader_gripper={summary['leader_gripper_updates']}, "
              f"physical_gripper={summary['gripper_updates']}, passed={summary['passed']}")
    return 0 if all(item['passed'] for item in report['channels'].values()) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs=2, default=['can0', 'can1'])
    parser.add_argument('--firmware', choices=['default', 'v111', 'v112', 'v120', 'v121'], default='v121',
                        help='Local SDK maps v121 to Nero firmware 1.21 or newer')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--rate', type=float, default=20.)
    parser.add_argument('--report', help='JSON path inside the project; default: hardware_checks/reports/')
    parser.add_argument('--self-check', action='store_true', help='Import SDK/dependencies without connecting hardware')
    args = parser.parse_args()
    if args.channels[0] == args.channels[1] or not .1 <= args.seconds <= 300 or not 1 <= args.rate <= 100:
        parser.error('Use distinct channels, seconds in [0.1, 300], and rate in [1, 100]')
    if args.report:
        try:
            inside_project(args.report)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        raise SystemExit(run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'HARDWARE_READ_FAILED: {exc}', file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
