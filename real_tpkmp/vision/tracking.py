"""Standalone camera-coordinate tracking diagnostics, never a robot target producer."""
from .common import ROOT, checked, object_config, output_path, new_directory, read_capture, write_json, digest, utc_now
from .capture import RGBDCamera
from .geometry import load_mesh, bounds, quality, render_depth
from .estimate import overlay
from .segmentation import hsv_masks
from .tracking_core import TrackingGuard, Measurements, center_position
from .box_prompt import MovingBox, checked_box, select_box
import concurrent.futures
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import numpy as np


def worker_environment():
    environment = os.environ.copy()
    toolchain = Path(environment.get('CUDA_HOME', ROOT/'envs/cuda_toolchain'))
    if (toolchain/'bin/nvcc').is_file():
        environment['CUDA_HOME'] = str(toolchain)
        environment['PATH'] = str(toolchain/'bin')+os.pathsep+environment.get('PATH', '')
        for key, paths in (('CPATH', [toolchain/'targets/x86_64-linux/include']),
                           ('LIBRARY_PATH', [toolchain/'targets/x86_64-linux/lib']),
                           ('LD_LIBRARY_PATH', [toolchain/'lib', toolchain/'targets/x86_64-linux/lib'])):
            environment[key] = os.pathsep.join(map(str, paths))+os.pathsep+environment.get(key, '')
    environment.setdefault('MAX_JOBS', '2')
    return environment


class Worker:
    def __init__(self, python, role, configuration, directory):
        if not Path(python).is_file():
            raise ValueError(f'找不到 {role} 的 Python：{python}；先检查 doctor.sh')
        self.log_path = directory/f'{role}.log'
        self.log = self.log_path.open('xb', buffering=0)
        self.process = subprocess.Popen([python, '-u', '-B', str(ROOT/'tracking_worker.py'),
                         '--role', role, '--config', str(configuration)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=worker_environment())
        self.buffer = b''

    def receive(self):
        deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                try:
                    result = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    self.log.write(line+b'\n')
                    continue
                if isinstance(result, dict) and 'ok' in result:
                    if not result['ok'] and result.get('fatal'):
                        raise RuntimeError(f'{result["error"]}；详见 {self.log_path}')
                    return result
            readable, _, _ = select.select([self.process.stdout], [], [], .2)
            if readable:
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError(f'GPU worker 已退出；详见 {self.log_path}')
                self.buffer += block
            elif self.process.poll() is not None:
                raise RuntimeError(f'GPU worker 已退出；详见 {self.log_path}')
        raise RuntimeError(f'GPU worker 超时；详见 {self.log_path}')

    def request(self, payload):
        self.process.stdin.write((json.dumps(payload)+'\n').encode())
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.log):
            stream.close()


def resize_frame(rgb, depth, K, max_width):
    import cv2
    h, w = depth.shape
    if w <= max_width:
        return rgb.copy(), depth.copy(), np.asarray(K).copy()
    size = (max_width, max(1, int(round(h*max_width/w))))
    scaled_K = np.diag([size[0]/w, size[1]/h, 1.])@K
    # Inputs are already registered COLOR pixels, so resize BOTH with this K.
    return (cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
            cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST), scaled_K)


def draw_processed(frame, result, triangles, cfg):
    import cv2
    rgb, depth, K = frame['rgb'], frame['depth_m'], frame['K']
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = result.get('_mask')
    if mask is not None:
        image[mask] = (image[mask]*.75+np.array([0, 255, 0])*.25).astype(np.uint8)
    if result.get('box_xyxy') is not None:
        h, w = depth.shape
        x0, y0, x1, y1 = np.rint(np.array(result['box_xyxy'])*[w, h, w, h]).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (230, 60, 230), 2)
    if result['T_color_object'] is not None:
        pose = checked(result['T_color_object'])
        image = overlay(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), K, pose, bounds(triangles),
                        'PASS' if result['accepted'] else 'LOST / REJECTED')
        h, w = depth.shape
        size = (min(320, w), max(1, int(round(h*min(320, w)/w))))
        small_K = np.diag([size[0]/w, size[1]/h, 1.])@K
        rendered = render_depth(triangles, pose, small_K, (size[1], size[0]))
        contours, _ = cv2.findContours((rendered > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 220, 255) if result['accepted'] else (0, 0, 255)
        contours = [np.rint(c*np.array([w/size[0], h/size[1]])).astype(np.int32) for c in contours]
        cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)
    return image


class Engine:
    def __init__(self, args, cfg, output):
        self.args, self.cfg, self.output = args, cfg, output
        self.triangles = load_mesh(cfg)
        self.center = bounds(self.triangles).mean(axis=0)
        self.guard = TrackingGuard(self.center, args.max_step_mm/1000, np.deg2rad(args.max_step_deg), args.max_gap_s)
        self.reset_requested = threading.Event()
        self.box_mode = getattr(args, 'mask_method', 'hsv') == 'box'
        self.box_state = MovingBox(getattr(args, 'box', None))
        self.needs_selection = threading.Event()
        if self.box_mode and self.box_state.box is None:
            self.needs_selection.set()
        self.workers, self.lock = [], threading.Lock()
        self.closed = False
        self.ready = False
        ipc = output/'.ipc'
        ipc.mkdir()
        self.input, self.mask_path = ipc/'frame.npz', ipc/'mask.npy'
        self.worker_config = output/'worker_config.json'
        values = dict(mesh_path=cfg['mesh_path'], mesh_scale_to_m=.001 if cfg['mesh_units'] == 'mm' else 1.,
                 mesh_rgb=cfg.get('mesh_rgb'), foundationpose_root=str(Path(args.foundationpose_root or ROOT/'third_party/FoundationPose').resolve()),
                 register_iterations=args.register_iterations, track_iterations=args.track_iterations,
                 debug_dir=str(output/'foundationpose_debug'),
                 sam3_source=str(Path(args.sam3_source or ROOT/'third_party/sam3').resolve()),
                 sam3_checkpoint=str(Path(args.sam3_checkpoint or ROOT/'weights/sam3.pt').resolve()),
                 threshold=cfg['min_segmentation_score'], box_prompt_mode=self.box_mode,
                 prompt=None if self.box_mode else args.prompt or cfg['text_prompt'], instance=args.instance)
        write_json(self.worker_config, values)

    def worker(self, python, role):
        with self.lock:
            if self.closed:
                raise RuntimeError('视觉 GPU 加载已取消')
            worker = Worker(python, role, self.worker_config, self.output)
            self.workers.append(worker)
        ready = worker.receive()
        if not ready.get('ready'):
            raise RuntimeError(f'{role} worker 没有完成初始化')
        return worker

    def start(self):
        print('加载常驻 GPU 模型；不会打开 CAN。首次运行可能编译 CUDA 内核。', flush=True)
        self.pose = self.worker(self.args.pose_python or os.environ.get('TPKMP_POSE_PYTHON', str(ROOT/'envs/foundationpose/bin/python')), 'pose')
        self.sam3 = None
        if self.args.mask_method in ('sam3', 'box'):
            self.sam3 = self.worker(self.args.sam3_python or os.environ.get('SAM3_PYTHON', str(ROOT/'envs/sam3/bin/python')), 'sam3')
        self.ready = True
        print('模型加载完成：首次配准，后续 track_one；每个处理帧重新分割并检查几何。', flush=True)

    def process(self, frame):
        start = time.monotonic()
        if self.reset_requested.is_set():
            self.reset_requested.clear()
            self.guard.reset()
            if self.box_mode:
                self.box_state.clear()
                self.needs_selection.set()
        if self.box_mode and frame.get('box_xyxy') is not None:
            self.box_state.select(frame['box_xyxy'])
            self.guard.reset()
            self.needs_selection.clear()
        result = dict(frame_number=frame['meta']['frame_number'], frame_meta=frame['meta'],
                      revision=frame['revision'], accepted=False, epoch=self.guard.epoch,
                      T_color_object=None, mode='lost', quality=None, reasons=[],
                      image_shape_hw=list(frame['depth_m'].shape), K=frame['K'].tolist(),
                      object_frame='original_mesh_scaled_to_m',
                      usable_for_robot=False, diagnostic_only=True)
        mask_validated = False
        try:
            with self.input.open('wb') as stream:
                np.savez(stream, rgb=frame['rgb'], depth_m=frame['depth_m'], K=frame['K'])
            if self.sam3:
                prompt_box = self.box_state.predict(frame['rgb']).tolist() if self.box_mode else None
                result['box_xyxy'] = prompt_box
                response = self.sam3.request(dict(input=str(self.input), output=str(self.mask_path), box_xyxy=prompt_box))
                if not response['ok']:
                    raise ValueError(response['error'])
                mask = np.load(self.mask_path, allow_pickle=False).astype(bool)
                result['segmentation_score'] = response['score']
            else:
                mask = hsv_masks(frame['rgb'][None], self.cfg, self.args.instance)[0][0]
                np.save(self.mask_path, mask, allow_pickle=False)
            if mask.shape != frame['depth_m'].shape:
                raise ValueError('分割 mask 与当前帧不匹配')
            result['_mask'] = mask  # Also show the mask when later depth checks reject it.
            valid = mask & (frame['depth_m'] >= self.cfg['min_depth_m']) & (frame['depth_m'] <= self.cfg['max_depth_m'])
            if np.count_nonzero(mask) < self.cfg['min_mask_pixels']:
                raise ValueError('目标太小')
            if np.count_nonzero(valid)/np.count_nonzero(mask) < self.cfg['min_mask_depth_fraction']:
                raise ValueError('目标有效深度不足')
            mask_validated = True
            if self.box_mode:
                self.box_state.accept(frame['rgb'], mask, prompt_box)
            response = self.pose.request(dict(input=str(self.input), mask=str(self.mask_path),
                                reset=self.guard.needs_registration(frame['meta']['host_monotonic_s'])))
            if not response['ok']:
                raise ValueError(response['error'])
            pose = checked(response['T_color_object'])
            report = quality(self.triangles, pose, frame['depth_m'], mask, frame['K'], self.cfg)
            result.update(self.guard.assess(pose, report, frame['meta']['host_monotonic_s'], response['registered']))
            result.update(T_color_object=pose.tolist(), center_color_m=center_position(pose, self.center).tolist(),
                          mode='register' if response['registered'] else 'track', quality=report,
                          network_pose_ms=response['elapsed_ms'])
        except ValueError as exc:
            self.guard.reset()
            result['reasons'] = [str(exc)]
            if self.box_mode and not mask_validated:
                self.box_state.clear()
                self.needs_selection.set()
        result['needs_box_selection'] = self.box_mode and self.needs_selection.is_set()
        result['segmentation_validated'] = mask_validated
        result['_view'] = draw_processed(frame, result, self.triangles, self.cfg)
        result['processing_ms'] = (time.monotonic()-start)*1000
        result['input_to_result_ms'] = (time.monotonic()-frame['received_monotonic_s'])*1000
        result['result_monotonic_s'] = time.monotonic()
        result['_frame'] = frame
        return result

    def close(self):
        with self.lock:
            self.closed = True
            for worker in self.workers:
                worker.close()


class Session:
    def __init__(self, output, args, center):
        self.output, self.args = output, args
        self.frames = (output/'frames.jsonl').open('x', encoding='utf-8')
        self.reports = (output/'measurements.jsonl').open('x', encoding='utf-8')
        self.measurements = Measurements(center, args.measure_samples, args.expected_distance_mm)
        self.count = self.accepted = self.snapshots = self.registered = self.discarded = 0
        self.selections = 0
        self.last = None
        self.history = []
        self.status_time = 0.
        if args.measure_static:
            self.measurements.start('static')

    def record(self, result):
        import cv2
        self.count += 1
        self.accepted += int(result['accepted'])
        self.registered += int(result['mode'] == 'register')
        value = {k: v for k, v in result.items() if not k.startswith('_')}
        self.frames.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        self.frames.flush()
        report = self.measurements.consume(result)
        if report:
            self.reports.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+'\n')
            self.reports.flush()
            print('测量结果：'+json.dumps(report, ensure_ascii=False), flush=True)
        self.last = result
        self.history.append(dict(time_s=result['frame_meta']['host_monotonic_s'],
                                 xyz=result.get('center_color_m') if result['accepted'] else None,
                                 quality=result['quality'], accepted=result['accepted']))
        cv2.imwrite(str(self.output/'latest.png'), result['_view'])
        rgb = result['_frame']['rgb']
        mask = result.get('_mask', np.zeros(rgb.shape[:2], bool))
        segmentation = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        segmentation[mask] = (segmentation[mask]*.5+np.array([0, 255, 0])*.5).astype(np.uint8)
        if '_mask' not in result:
            cv2.putText(segmentation, 'NO VALID MASK', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        cv2.imwrite(str(self.output/'mask.png'), mask.astype(np.uint8)*255)
        cv2.imwrite(str(self.output/'segmentation.png'), segmentation)
        if self.args.save_every > 0 and self.count % self.args.save_every == 0:
            self.snapshot()
        if time.monotonic()-self.status_time > 1.:
            q = result['quality']
            print(f'{result["mode"]} accepted={result["accepted"]} | {result["processing_ms"]:.0f} ms | '
                  f'IoU={q["render_mask_iou"]:.3f}' if q else f'LOST: {result["reasons"]}', flush=True)
            self.status_time = time.monotonic()

    def selection(self, frame, box):
        import cv2
        self.selections += 1
        directory = self.output/'selections'
        directory.mkdir(exist_ok=True)
        stem = directory/f'{self.selections:04d}'
        write_json(stem.with_suffix('.json'), dict(box_xyxy=checked_box(box).tolist(),
                   frame_meta=frame['meta'], K=frame['K'].tolist(), image_shape_hw=list(frame['depth_m'].shape)))
        image = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
        h, w = frame['depth_m'].shape
        x0, y0, x1, y1 = np.rint(np.asarray(box)*[w, h, w, h]).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (230, 60, 230), 2)
        cv2.imwrite(str(stem.with_suffix('.png')), image)

    def snapshot(self):
        import cv2
        if self.last is None:
            return
        self.snapshots += 1
        directory = self.output/'snapshots'
        directory.mkdir(exist_ok=True)
        stem = directory/f'{self.snapshots:04d}'
        cv2.imwrite(str(stem.with_suffix('.png')), self.last['_view'])
        frame = self.last['_frame']
        np.savez_compressed(stem.with_suffix('.npz'), rgb=frame['rgb'], depth_m=frame['depth_m'], K=frame['K'],
                            mask=self.last.get('_mask', np.zeros(frame['depth_m'].shape, bool)))
        write_json(stem.with_suffix('.json'), {k: v for k, v in self.last.items() if not k.startswith('_')})
        print(f'保存当前已处理帧：{stem}', flush=True)

    def close(self, error=None):
        self.frames.close()
        self.reports.close()
        plot_error = None
        if self.history:
            try:
                plot_history(self.history, self.output/'tracking.png')
            except Exception as exc:
                plot_error = str(exc)
                print(f'轨迹图未生成：{exc}；逐帧 JSONL 已保存。')
        write_json(self.output/'summary.json', dict(schema_version=1, kind='vision_tracking_summary',
            processed_frames=self.count, accepted_frames=self.accepted, rejected_frames=self.count-self.accepted,
            discarded_on_manual_reset=self.discarded, registration_calls=self.registered, snapshots=self.snapshots,
            box_selections=self.selections,
            error=error, plot_error=plot_error, opens_camera=not bool(self.args.replay),
            opens_can=False, usable_for_robot=False, diagnostic_only=True,
            absolute_pose_accuracy_tested=False, note='Jitter and RGB-D agreement do not establish absolute pose accuracy.'))


def plot_history(history, destination):
    import matplotlib.pyplot as plt
    t = np.array([r['time_s'] for r in history])
    t -= t[0]
    xyz = np.array([r['xyz'] if r['xyz'] is not None else [np.nan]*3 for r in history])*1000
    valid = np.isfinite(xyz).all(axis=1)
    if np.any(valid):
        xyz -= xyz[np.flatnonzero(valid)[0]].copy()
    iou = [r['quality']['render_mask_iou'] if r['quality'] else np.nan for r in history]
    depth = [r['quality']['depth_median_error_m']*1000
             if r['quality'] and r['quality']['depth_median_error_m'] is not None else np.nan for r in history]
    lost = np.array([not r['accepted'] for r in history])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for column, label in enumerate(('X', 'Y', 'Z')):
        axes[0].plot(t, xyz[:, column], '.-', label=label)
    axes[0].set(ylabel='Center change (mm)', title='Relative to first accepted center in optical frame; gaps are rejected poses')
    axes[0].legend()
    axes[1].plot(t, iou, '.-')
    axes[1].set(ylabel='Mask / CAD IoU', ylim=(0, 1.05))
    axes[2].plot(t, depth, '.-')
    axes[2].set(ylabel='Depth median error (mm)', xlabel='Input stream time (s)')
    for axis in axes:
        for stamp in t[lost]:
            axis.axvline(stamp, color='red', alpha=.35)
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def frame_input(rgb, depth, K, metadata, args, revision=0):
    rgb, depth, K = resize_frame(rgb, depth, K, args.max_width)
    return dict(rgb=rgb, depth_m=depth, K=K, meta=metadata, revision=revision,
                received_monotonic_s=time.monotonic())


def check_display(args):
    if args.headless:
        return
    import cv2
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise ValueError('当前没有桌面显示；在桌面终端运行，或加 --headless 保存结果')
    if 'GUI:                           NONE' in cv2.getBuildInformation():
        raise ValueError('当前 OpenCV 不支持窗口；使用现有 tpkmp 的 .sh 入口，或加 --headless')
    cv2.namedWindow('Vision only - no CAN', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vision only - no CAN', 1120, 660)


def dashboard(rgb, session, engine, replay=False):
    import cv2
    width = 560
    h, w = rgb.shape[:2]
    height = int(round(h*width/w))
    canvas = np.full((height+260, width*2, 3), 25, np.uint8)
    canvas[70:70+height, :width] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height))
    last = session.last
    if last:
        canvas[70:70+height, width:] = cv2.resize(last['_view'], (width, height))
    def line(text, x, y, color=(230, 230, 230)):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .53, color, 1, cv2.LINE_AA)
    line('REPLAY RGB' if replay else 'LIVE RGB (no old pose overlaid)', 15, 25)
    line('PROCESSED FRAME + CAD outline / axes', width+15, 25)
    line('Camera optical: X right, Y down, Z forward | red X, green Y, blue Z', 15, 52)
    y = height+95
    if not last:
        message = 'Loading GPU models...' if not engine.ready else 'Registering object...'
        if engine.ready and getattr(engine, 'box_mode', False) and engine.needs_selection.is_set():
            message = 'Press F to select target. Draw box, Enter confirm; keep object still.'
        line(message, 15, y, (0, 220, 255))
    else:
        color = (60, 230, 60) if last['accepted'] else (60, 60, 255)
        age = time.monotonic()-(last['result_monotonic_s'] if replay else last['frame_meta']['host_monotonic_s'])
        line(f'{last["mode"].upper()} | accepted={last["accepted"]} | frame={last["frame_number"]} | age={age:.2f}s', 15, y, color)
        line(f'Processing={last["processing_ms"]:.0f}ms | submitted-to-result={last["input_to_result_ms"]:.0f}ms', 15, y+24)
        q = last['quality']
        if q:
            median = q['depth_median_error_m']
            p90 = q['depth_p90_error_m']
            line(f'IoU={q["render_mask_iou"]:.3f} | depth median={median*1000:.2f}mm / p90={p90*1000:.2f}mm'
                 if median is not None and p90 is not None else 'Insufficient overlapping depth', 15, y+48)
        if last['accepted']:
            xyz = np.array(last['center_color_m'])*1000
            line(f'Mesh CENTER XYZ (mm) = {xyz.round(2).tolist()} | epoch={last["epoch"]}', 15, y+72)
        else:
            line(('LOST: '+', '.join(last['reasons']))[:145], 15, y+72, color)
            if last.get('needs_box_selection'):
                line('Press F to reselect.', 850, y+72, color)
    line(session.measurements.message[:145], 15, y+104, (0, 220, 255))
    line('F reselect | S static | A/B translation | R reset/reselect | P snapshot | Q quit', 15, y+130)
    line('DIAGNOSTIC ONLY: no CAN / hand-eye / robot motion. Small jitter does not prove accuracy.', 15, y+155)
    cv2.imshow('Vision only - no CAN', canvas)
    if cv2.getWindowProperty('Vision only - no CAN', cv2.WND_PROP_VISIBLE) < 1:
        return ord('q')
    return cv2.waitKey(1) & 255


def key_action(key, session, engine):
    if key in (ord('q'), 27):
        return 'quit'
    if key == ord('r') or (key == ord('f') and getattr(engine, 'box_mode', False)):
        engine.reset_requested.set()
        if getattr(engine, 'box_mode', False):
            engine.needs_selection.set()
        session.measurements.cancel('manual reset', invalidate_anchor=True)
        session.last = None
        return 'reset'
    if key == ord('p'):
        session.snapshot()
    for char, kind in (('s', 'static'), ('a', 'A'), ('b', 'B')):
        if key == ord(char):
            session.measurements.start(kind)
    return None


def track(args):
    import cv2
    cfg = object_config(args.object)
    load_mesh(cfg)  # Validate units/hash before starting devices or large workers.
    if args.mask_method == 'hsv' and not all(k in cfg for k in ('hsv_lower', 'hsv_upper')):
        raise ValueError('这个模型没有 HSV 配置，请使用 --mask-method sam3')
    check_display(args)
    output = new_directory(output_path(args.out))
    engine = Engine(args, cfg, output)
    session = Session(output, args, engine.center)
    camera = executor = None
    error = None
    started = time.monotonic()
    try:
        if args.replay:
            directory, metadata, data = read_capture(args.replay)
            write_json(output/'session.json', dict(schema_version=1, kind='vision_tracking_session',
                       created_utc=utc_now(), object=cfg, object_config_sha256=digest(args.object or ROOT/'objects/keel.json'),
                       mesh_sha256=digest(cfg['mesh_path']), replay=str(directory), synthetic=metadata['synthetic'],
                       opens_camera=False, opens_can=False, diagnostic_only=True, usable_for_robot=False,
                       camera=metadata['camera'], segmentation_mode=args.mask_method,
                       initial_box=getattr(args, 'box', None), prompt=None if engine.box_mode else args.prompt or cfg['text_prompt']))
            engine.start()
            for i, (rgb, depth, meta) in enumerate(zip(data['rgb'], data['depth_m'], metadata['frames'])):
                if args.max_frames and session.count >= args.max_frames:
                    break
                # Replay uses a logical stream clock; latency is processing only.
                meta = dict(meta, host_monotonic_s=i/metadata['camera']['fps'], replay=True,
                            source_frame_number=meta['frame_number'])
                frame = frame_input(rgb, depth, data['K'], meta, args)
                if engine.box_mode and engine.needs_selection.is_set() and not args.headless:
                    selection = select_box(frame['rgb'])
                    if selection is None:
                        break
                    frame['box_xyxy'] = selection
                    session.selection(frame, selection)
                elif engine.box_mode and i == 0 and args.box is not None:
                    frame['box_xyxy'] = args.box
                    session.selection(frame, args.box)
                session.record(engine.process(frame))
                if not args.headless and key_action(dashboard(rgb, session, engine, True), session, engine) == 'quit':
                    break
        else:
            camera = RGBDCamera(args.serial, args.width, args.height, args.fps, args.intrinsics)
            write_json(output/'session.json', dict(schema_version=1, kind='vision_tracking_session',
                       created_utc=utc_now(), object=cfg, object_config_sha256=digest(args.object or ROOT/'objects/keel.json'),
                       mesh_sha256=digest(cfg['mesh_path']), synthetic=False, opens_camera=True, opens_can=False,
                       diagnostic_only=True, usable_for_robot=False, camera=camera.model,
                       source_camera=camera.source_model, intrinsics_sha256=camera.intrinsics_sha256,
                       segmentation_mode=args.mask_method, initial_box=getattr(args, 'box', None),
                       prompt=None if engine.box_mode else args.prompt or cfg['text_prompt']))
            print('只打开 D435i，不读 CAN、不使用手眼结果。固定相机，缓慢移动物体。', flush=True)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            warmup = executor.submit(engine.start)
            pending, revision = None, 0
            selected_frame = None
            selection_requested = waiting_for_selection = engine.box_mode and args.box is None
            while True:
                rgb, depth, meta = camera.frame()
                if warmup.done():
                    warmup.result()
                if pending is not None and pending.done():
                    result = pending.result()
                    pending = None
                    if result['revision'] == revision:
                        session.record(result)
                    else:
                        session.discarded += 1
                if args.max_frames and session.count >= args.max_frames:
                    break
                if args.seconds and time.monotonic()-started >= args.seconds:
                    break
                action = None if args.headless else key_action(dashboard(rgb, session, engine), session, engine)
                if action == 'quit':
                    break
                if action == 'reset':
                    revision += 1
                    if engine.box_mode:
                        selection_requested = waiting_for_selection = True
                if engine.ready and pending is None and selection_requested:
                    frozen = frame_input(rgb, depth, camera.K, meta, args, revision)
                    selection = select_box(frozen['rgb'])
                    selection_requested = False
                    if selection is not None:
                        revision += 1
                        frozen.update(box_xyxy=selection, revision=revision)
                        selected_frame = frozen
                        waiting_for_selection = False
                        session.selection(frozen, selection)
                        session.measurements.message = 'Box selected. Keep object still until first registration finishes.'
                    else:
                        waiting_for_selection = True
                        session.measurements.message = 'Selection canceled. Press F to draw another box, Q to quit.'
                if args.headless and engine.box_mode and engine.ready and pending is None and engine.needs_selection.is_set():
                    raise ValueError('目标丢失，需要重新框选；无窗口模式请重新指定 --box 后启动')
                can_dispatch = (not engine.box_mode or selected_frame is not None or
                                (not waiting_for_selection and not engine.needs_selection.is_set()))
                if engine.ready and pending is None and can_dispatch:
                    # At most one in flight: take the latest frame, never queue a backlog.
                    frame = selected_frame or frame_input(rgb, depth, camera.K, meta, args, revision)
                    selected_frame = None
                    if engine.box_mode and session.selections == 0 and args.box is not None:
                        frame['box_xyxy'] = args.box
                        session.selection(frame, args.box)
                    pending = executor.submit(engine.process, frame)
    except KeyboardInterrupt:
        print('用户停止视觉预览。')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        engine.close()
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        if camera:
            camera.close()
        if not args.headless:
            cv2.destroyAllWindows()
        session.close(error)
        print(f'视觉诊断输出：{output}；未连接 CAN，结果不能直接用于真机执行。', flush=True)
