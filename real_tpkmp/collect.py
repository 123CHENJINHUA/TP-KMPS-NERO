"""Physically teach the leader; record follower measured states without sending CAN."""
from .bootstrap import new_directory, write_json, utc_now, digest
import argparse
from pathlib import Path
import time
import numpy as np
from .data import config, physical_signature, save_record, align, STAGES, frame_document, read_frames
from .hardware import Reader
from .terminal import keys, flush_keys
from .vision.settings import DEFAULT_CONFIG


def stable_state(reader):
    samples = []
    for _ in range(15):
        samples.append(reader.sample())
        time.sleep(.02)
    if np.max(np.ptp([s['joints_rad'] for s in samples], axis=0)) > .01:
        raise ValueError('还在运动，请停稳后重新记录参考点')
    return samples[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config')
    parser.add_argument('--can', default=None)
    parser.add_argument('--demos', type=int, default=5)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--task-frames', help='可选视觉/手动参考系文件；R 时读取其中 pick/place，起点仍由实际状态记录')
    source.add_argument('--vision-config', '--camera-config', dest='vision_config',
                        help='覆盖默认 vision/camera_config.json；V 定位并预览，L 记录固定放置点，R 开始本轮')
    source.add_argument('--manual', action='store_true', help='人工 P/L 模式，不读取相机配置')
    parser.add_argument('--place-frames', help='仅视觉模式：从已有真实 frames.json 初始化固定 place；也可直接按 L')
    args = parser.parse_args()
    if args.demos < 1:
        parser.error('--demos 必须为正整数')
    if not args.manual and not args.task_frames:
        args.vision_config = args.vision_config or str(DEFAULT_CONFIG)
    if args.place_frames and not args.vision_config:
        parser.error('--place-frames 需要 --vision-config')
    cfg = config(args.config, args.can)
    settings = None
    if args.vision_config:
        from .visual_collection import load_settings, VisualCollection
        try:
            settings = load_settings(args.vision_config, cfg)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f'读取相机配置：{Path(args.vision_config).expanduser().resolve()}', flush=True)
    reader = Reader(cfg)
    try:
        reader.wait_ready()
        output = new_directory(args.out)
        visual = VisualCollection(settings, cfg, output, args.place_frames) if settings is not None else None
        write_json(output/'session.json', dict(started_utc=utc_now(), config=cfg, joint_source='measured',
                                              task='pick_place', task_success_source='operator',
                                              collection_mode='visual' if visual else 'task_frames' if args.task_frames else 'manual',
                                              visual_runs=str(visual.directory) if visual else None))
        print(f'输出 {output}\n只接收从臂反馈，不发运动指令。按键只标记记录；机械臂通过主臂拖动。')
        if visual:
            print('视觉模式：V 拍照定位，在预览窗口 Enter 接受/C 取消；L 记录固定放置 TCP（跨轮复用）。')
            print('保持物体不动，回统一起点后 R 开始；每轮都要重新 V。P 在视觉模式中不覆盖视觉目标。')
            print(f'视觉记录：{visual.directory}')
        elif args.task_frames:
            print('参考系文件模式：R 读取文件中的 pick/place，以当前实际 TCP 作为起点。')
        else:
            print('录制前：P 记录抓取参考 TCP；L 记录放置参考 TCP；回到统一起点后 R 开始。')
        print('N 依次切换：接近→夹取→抬升→搬运→松开→撤离；S 确认本次真实任务成功并保存。')
        print('X 丢弃当前尝试；Q 退出。记录参考点和拍照时均需停稳；物体改变位置后必须重新定位。')
        pick = place = None
        frame_evidence = {}
        frame_source = None
        rows, started, stage, mode, accepted = [], None, 0, None, 0
        last_status, last_warning, next_tick = 0., 0., time.monotonic()
        with keys() as key:
            while accepted < args.demos:
                now, pressed = time.monotonic(), key()
                if pressed in ('q', '\x1b'):
                    break
                try:
                    state = reader.sample()
                except ValueError as exc:
                    if started is not None:
                        rows, started = [], None
                        frame_evidence, frame_source = {}, None
                        if visual:
                            visual.invalidate()
                        print(f'本次已丢弃（反馈断档）：{exc}')
                    if now-last_warning > 2:
                        print(str(exc))
                        last_warning = now
                    time.sleep(.02)
                    continue
                if pressed == 'v':
                    if started is not None:
                        print('正在记录动作，不能重新定位；先 S 保存或 X 丢弃。')
                    elif visual is None:
                        print('当前为人工/文件模式；重新启动时省略 --manual/--task-frames 即可使用视觉采集。')
                    else:
                        try:
                            visual.localize(reader)
                        except (ValueError, RuntimeError, OSError) as exc:
                            visual.invalidate()
                            print(f'本轮定位未完成：{exc}；调整后按 V 重试。', flush=True)
                        finally:
                            flush_keys()
                            next_tick = time.monotonic()
                    if started is None:
                        continue
                if pressed in ('p', 'l') and started is None:
                    if visual and pressed == 'p':
                        print('视觉模式使用 V 计算 pick；P 不覆盖视觉目标。')
                        continue
                    if args.task_frames:
                        print('当前使用 --task-frames；参考点由文件提供。')
                        continue
                    try:
                        point = stable_state(reader)
                        pose = point['tcp'].copy()
                        if visual:
                            visual.set_place(pose, dict(type='live_measured', channel=cfg['channel'],
                                                       joints_rad=point['joints_rad'].tolist(), created_utc=utc_now()))
                    except (ValueError, OSError) as exc:
                        print(f'参考点未记录：{exc}')
                        continue
                    finally:
                        flush_keys()
                    if pressed == 'p':
                        pick = pose
                    else:
                        place = pose
                    print(f'{"抓取" if pressed == "p" else "放置"} TCP 已记录：{pose[:3]} m')
                if pressed == 'r' and started is None:
                    try:
                        frame_evidence, frame_source = {}, None
                        if visual:
                            frames, document, source_path, state = visual.start(reader)
                        elif args.task_frames:
                            from .vision.bindings import frame_reasons
                            document, tasks = read_frames(args.task_frames, cfg)
                            reasons = frame_reasons(document)
                            if document.get('synthetic', True):
                                reasons.append('synthetic_task_frames')
                            if reasons:
                                raise ValueError(f'参考系未通过检查：{reasons}')
                            pick, place = tasks[1].copy(), tasks[2].copy()
                            frames = np.array([state['tcp'], pick, place])
                            source_path = Path(args.task_frames).resolve()
                        else:
                            if pick is None or place is None:
                                raise ValueError('先用 P、L 记录两个参考 TCP')
                            frames = np.array([state['tcp'], pick, place])
                        if visual or args.task_frames:
                            frame_evidence = {name: document[name] for name in
                                ('vision', 'execution_blocked_reasons', 'place_reference', 'vision_collection') if name in document}
                            frame_source = dict(path=str(source_path), sha256=digest(source_path), synthetic=False, **frame_evidence)
                    except (ValueError, OSError) as exc:
                        print(f'未开始记录：{exc}')
                        continue
                    now = time.monotonic()
                    if now-state['host_monotonic_s'] > cfg['state_max_age_s']:
                        if visual:
                            visual.invalidate()
                        print('开始检查期间反馈已过期；重新准备本轮后按 R。')
                        continue
                    rows, started, stage, mode = [], now, 0, state['gripper_mode']
                    print('开始 1/6 approach；到抓取点、即将闭爪时按 N。')
                if pressed == 'x':
                    rows, started = [], None
                    frame_evidence, frame_source = {}, None
                    if visual:
                        visual.invalidate()
                    print('当前尝试已丢弃；'+('按 V 重新定位，再按 R 重试。' if visual else '调整后按 R 重试。'))
                if pressed == 'n' and started is not None and stage < 5:
                    stage += 1
                    print(f'{stage+1}/6 {STAGES[stage]}')
                if started is not None:
                    if state['gripper_mode'] != mode:
                        rows, started = [], None
                        frame_evidence, frame_source = {}, None
                        if visual:
                            visual.invalidate()
                        print('夹爪模式改变，本次丢弃。')
                        continue
                    rows.append((now-started, stage, state))
                if pressed == 's' and started is not None:
                    data = dict(time_s=np.array([r[0] for r in rows]), stage=np.array([r[1] for r in rows], int))
                    for name in ('tcp', 'joints_rad', 'gripper_measured', 'gripper_command', 'gripper_force_N', 'gripper_feedback_force_N'):
                        data[name] = np.array([r[2][name] for r in rows])
                    try:
                        align(data, 96, cfg)
                    except ValueError as exc:
                        print(f'未保存：{exc}；可继续记录或 X 重试。')
                        continue
                    accepted += 1
                    stem = output/f'demo_{accepted:03d}'
                    meta = dict(schema_version=1, robot='nero', kind='demonstration', synthetic=False,
                                created_utc=utc_now(), config=cfg, signature=physical_signature(cfg),
                                joint_source='measured', channel=cfg['channel'], gripper_mode=mode,
                                success=True, success_source='operator', task='pick_place')
                    if frame_source:
                        meta['task_frame_source'] = frame_source
                    save_record(stem.with_suffix('.npz'), meta, frames=frames, **data)
                    write_json(output/f'demo_{accepted:03d}_frames.json', frame_document(frames, cfg, **frame_evidence))
                    print(f'已保存成功示教 {accepted}/{args.demos}：{stem.name}；'+
                          ('下一组重新 V/R；固定放置点已保留，可按 L 更新。' if visual else '下一组重新准备参考点后 R。'))
                    rows, started, pick, place = [], None, None, None
                    frame_evidence, frame_source = {}, None
                now = time.monotonic()
                if now-last_status > 2:
                    print(f'{"记录中" if started is not None else "待机"} | q(deg)={np.round(np.rad2deg(state["joints_rad"]), 1)} | '
                          f'gripper={state["gripper_measured"]:.4f} ({state["gripper_mode"]})')
                    last_status = now
                next_tick = max(next_tick+1/cfg['record_hz'], time.monotonic())
                time.sleep(max(0, next_tick-time.monotonic()))
        print(f'结束：保存 {accepted} 组成功示教。未保存的尝试不会进入训练。')
    except KeyboardInterrupt:
        print('用户停止采集。未保存的尝试不会进入训练。')
    finally:
        reader.close()


if __name__ == '__main__':
    main()
