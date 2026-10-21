"""Capture start/pick/place TCP frames passively, or copy/offset an existing set."""
from .bootstrap import write_json
import argparse
import time
import numpy as np
from .data import config, read_frames, frame_document
from .hardware import Reader
from .terminal import keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config')
    parser.add_argument('--can')
    parser.add_argument('--from-frames')
    parser.add_argument('--pick-offset-m', nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument('--place-offset-m', nargs=3, type=float, default=[0, 0, 0])
    args = parser.parse_args()
    cfg = config(args.config, args.can)
    if args.from_frames:
        meta, frames = read_frames(args.from_frames, cfg)
        frames[1, :3] += args.pick_offset_m
        frames[2, :3] += args.place_offset_m
        evidence = {key: meta[key] for key in ('vision', 'execution_blocked_reasons') if key in meta}
        write_json(args.out, frame_document(frames, cfg, meta.get('synthetic', True), source=args.from_frames, **evidence))
    else:
        if np.any(args.pick_offset_m) or np.any(args.place_offset_m):
            raise ValueError('位移参数仅与 --from-frames 一起使用')
        reader, frames = Reader(cfg), {}
        try:
            reader.wait_ready(need_command=False)
            print('拖动主臂定位并停稳：S 起点，P 抓取 TCP，L 放置 TCP；W 写入；Q 退出。只接收 CAN。')
            with keys() as key:
                while True:
                    pressed = key()
                    if pressed in ('q', '\x1b'):
                        return
                    if pressed in ('s', 'p', 'l'):
                        states = []
                        for _ in range(15):
                            states.append(reader.sample(False))
                            time.sleep(.02)
                        if np.max(np.ptp([s['joints_rad'] for s in states], axis=0)) > .01:
                            print('请停稳再记录。')
                            continue
                        frames[{'s': 'start', 'p': 'pick', 'l': 'place'}[pressed]] = states[-1]['tcp']
                        print(f'已记录：{list(frames)}')
                    if pressed == 'w':
                        if set(frames) != {'start', 'pick', 'place'}:
                            print('需要三个参考点。')
                            continue
                        write_json(args.out, frame_document([frames[n] for n in ('start', 'pick', 'place')], cfg))
                        break
                    time.sleep(.02)
        finally:
            reader.close()
    print(args.out)


if __name__ == '__main__':
    main()
