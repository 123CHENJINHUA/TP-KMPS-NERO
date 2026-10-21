"""Passively check two SocketCAN channels; no third-party modules or CAN transmit."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import selectors
import socket
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
CAN_FRAME = struct.Struct('=IB3x8s')
CAN_EFF_MASK = 0x1FFFFFFF
CAN_SFF_MASK = 0x7FF
CAN_EFF_FLAG = 0x80000000
CAN_ERR_FLAG = 0x20000000


def parse_frame(data):
    if len(data) < CAN_FRAME.size:
        raise ValueError(f'Incomplete CAN frame: {len(data)} bytes')
    can_id, length, _ = CAN_FRAME.unpack_from(data)
    identity = can_id & (CAN_EFF_MASK if can_id & CAN_EFF_FLAG else CAN_SFF_MASK)
    return identity, min(length, 8), bool(can_id & CAN_ERR_FLAG)


def inside_project(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Report path must be inside this project')
    return result


def probe(channels, seconds):
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'duration_s': seconds, 'channels': {}}
    selector = selectors.DefaultSelector()
    sockets = []
    for channel in channels:
        result = {'frames': 0, 'error_frames': 0, 'ids': Counter(), 'opened': False}
        report['channels'][channel] = result
        try:
            flags = int((Path('/sys/class/net') / channel / 'flags').read_text().strip(), 16)
            if not flags & 1:
                raise OSError(f'{channel} exists but is DOWN')
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            try:
                sock.setblocking(False)
                sock.bind((channel,))
            except BaseException:
                sock.close()
                raise
            result['opened'] = True
            selector.register(sock, selectors.EVENT_READ, result)
            sockets.append(sock)
        except FileNotFoundError:
            result['error'] = f'{channel} not found; connect and configure its SocketCAN adapter'
        except (OSError, ValueError) as exc:
            result['error'] = str(exc)
    deadline = time.monotonic() + seconds
    try:
        while selector.get_map() and time.monotonic() < deadline:
            for key, _ in selector.select(max(0., min(.2, deadline - time.monotonic()))):
                channel_report = key.data
                try:
                    data = key.fileobj.recv(72)
                    can_id, length, error = parse_frame(data)
                    channel_report['frames'] += 1
                    channel_report['error_frames'] += int(error)
                    channel_report['ids'][f'0x{can_id:03X}'] += 1
                except (OSError, ValueError) as exc:
                    channel_report['error'] = str(exc)
                    selector.unregister(key.fileobj)
    finally:
        selector.close()
        for sock in sockets:
            sock.close()
    for result in report['channels'].values():
        result['ids'] = dict(result['ids'].most_common(12))
        result['frames_per_second'] = round(result['frames'] / seconds, 1)
        result['passed'] = bool(result['frames'] > 0 and not result.get('error'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs=2, default=['can0', 'can1'])
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--report', help='Optional JSON path inside the project')
    args = parser.parse_args()
    if not 0.1 <= args.seconds <= 300 or args.channels[0] == args.channels[1]:
        parser.error('Use distinct channels and --seconds in [0.1, 300]')
    try:
        report_path = inside_project(args.report) if args.report else None
    except ValueError as exc:
        parser.error(str(exc))
    report = probe(args.channels, args.seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    if not all(item['passed'] for item in report['channels'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
