"""Receive-only live motion monitor for matching a physical arm to CAN feedback."""
from bootstrap import HERE, inside_folder, write_json, utc_now
import argparse
from datetime import datetime
import time
import numpy as np
from robot_pose import CanPoseReader
from transforms import distance


def summarize(samples):
    if not samples:
        return {'valid_samples': 0}
    joints = np.array([s['joints_rad'] for s in samples])
    deltas = [distance(samples[0]['T_base_flange'], s['T_base_flange']) for s in samples]
    return {'valid_samples': len(samples), 'joint_range_deg': np.rad2deg(np.ptp(joints, axis=0)).tolist(),
            'max_flange_translation_mm': max(v[0] for v in deltas) * 1000.,
            'max_flange_rotation_deg': float(np.rad2deg(max(v[1] for v in deltas))),
            'joint_timestamp_advance_s': (np.array(samples[-1]['joint_frame_unix_s']) - np.array(samples[0]['joint_frame_unix_s'])).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channels', nargs='+', choices=['can0', 'can1'], default=['can0', 'can1'])
    p.add_argument('--joint-source', choices=['measured', 'leader', 'both'], default='both')
    p.add_argument('--firmware', default='v121')
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--report-dir', help='默认 diagnostics/motion_<时间戳>')
    args = p.parse_args()
    if not .5 <= args.seconds <= 300 or len(set(args.channels)) != len(args.channels):
        p.error('时长必须在 0.5..300 秒，通道不能重复')
    out = inside_folder(args.report_dir or HERE / 'diagnostics' / ('motion_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if out.exists():
        raise ValueError('报告目录已存在，请更换名称')
    sources = ('measured', 'leader') if args.joint_source == 'both' else (args.joint_source,)
    readers, channels = {}, {}
    report = {'created_utc': utc_now(), 'read_only': True, 'firmware': args.firmware, 'channels': channels,
              'note': '请在监听期间让装相机或标定板的实体臂运动；有新时间戳但角度不变不能证明运动反馈正常。'
                      '主从联动时两种来源都可能变化，从臂标定需选择其 measured 实际反馈，不能用 leader 指令代替。'}
    try:
        for channel in args.channels:
            try:
                readers[channel] = CanPoseReader(channel, args.firmware, sources[0])
                channels[channel] = {'sources': {s: {'samples': [], 'last_error': None} for s in sources}}
            except (ValueError, OSError) as exc:
                channels[channel] = {'error': str(exc)}
                print(f'{channel}: {exc}', flush=True)
        print(f'只接收 CAN。接下来 {args.seconds:g} 秒，请让装相机或标定板的实体臂运动；'
              '观察哪个通道的 q(deg) 随动作变化。', flush=True)
        print('主从联动时 leader 和 measured 都可能变化；相机/板装在从臂时选其 measured 实际反馈。'
              '共用 CAN 总线时，两种来源可以出现在同一通道。', flush=True)
        start, next_print = time.monotonic(), 0.
        while readers and time.monotonic() - start < args.seconds:
            elapsed = time.monotonic() - start
            statuses = {channel: reader.inspect_sources() for channel, reader in readers.items()}
            for channel, status in statuses.items():
                for source in sources:
                    value = channels[channel]['sources'][source]
                    entry = status[source]
                    if entry['available']:
                        value['samples'].append(entry['packet'])
                        value['last_error'] = None
                    else:
                        value['last_error'] = entry['error']
            if elapsed >= next_print:
                for channel, status in statuses.items():
                    transport = readers[channel].transport_status()
                    for source in sources:
                        entry = status[source]
                        if not entry['available']:
                            print(f'{elapsed:5.1f}s {channel}/{source}: 无完整新鲜反馈'
                                  f'（原始 CAN 累计 {transport["frames"]} 帧）', flush=True)
                            continue
                        packet = entry['packet']
                        base = channels[channel]['sources'][source]['samples'][0]
                        t, r = distance(base['T_base_flange'], packet['T_base_flange'])
                        q = ' '.join(f'{v:8.3f}' for v in np.rad2deg(packet['joints_rad']))
                        print(f'{elapsed:5.1f}s {channel}/{source} q(deg)=[{q}]  法兰Δ={t*1000:.2f}mm/{np.rad2deg(r):.2f}deg', flush=True)
                next_print = elapsed + .5
            time.sleep(.025)
    except KeyboardInterrupt:
        report['interrupted'] = True
    finally:
        for channel, reader in readers.items():
            channels[channel]['transport'] = reader.transport_status()
            reader.close()
    any_samples = False
    for channel, value in channels.items():
        transport = value.get('transport')
        if transport is not None:
            print(f'{channel} 原始 CAN: {transport["frames"]} 帧，ID={transport["ids"]}', flush=True)
            if transport['receiver_error']:
                print(f'{channel}: 接收失败：{transport["receiver_error"]}', flush=True)
            elif not transport['frames']:
                print(f'{channel}: 没有收到原始报文。先核对这路适配器的 CAN 端接线、机械臂供电和广播；'
                      '零报文不能靠更换关节解析来源解决。', flush=True)
        for source, data in value.get('sources', {}).items():
            data['summary'] = summarize(data['samples'])
            any_samples |= bool(data['samples'])
            print(f'{channel}/{source} 摘要: {data["summary"]}', flush=True)
    write_json(out / 'report.json', report)
    print(f'报告: {out / "report.json"}', flush=True)
    return 0 if any_samples else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
