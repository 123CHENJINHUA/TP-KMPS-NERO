"""Manual static-position acquisition; SPACE saves a fresh synchronized static sample."""
from bootstrap import HERE, inside_folder, write_json, utc_now
import argparse
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from vision import load_board, BoardDetector
from realsense_camera import RealSenseCamera
from robot_pose import CanPoseReader, manual_pose, require_stationary
from motion_checks import check_capture, compare_capture, MotionCaptureError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['eye-in-hand', 'eye-to-hand', 'intrinsics'])
    p.add_argument('--out', default=str(HERE / 'datasets' / datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--board-config', default=str(HERE / 'board_config.json'))
    p.add_argument('--dictionary')
    p.add_argument('--serial')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--intrinsics', help='可选：calibrate_intrinsics.py 输出的 RGB 内参 JSON')
    p.add_argument('--with-depth', action='store_true', help='额外保存 RGB/Depth 外参；手眼仍使用 RGB')
    p.add_argument('--pose-source', choices=['can', 'manual'], default='can')
    p.add_argument('--can', choices=['can0', 'can1'], default='can0')
    p.add_argument('--firmware', default='v121')
    p.add_argument('--joint-source', choices=['measured', 'leader'], default='measured',
                   help='measured=实际关节反馈；leader=主臂角度/从臂指令，相机或板装在从臂时必须用 measured')
    p.add_argument('--pose-file', help='manual 模式：每次采样读取此 JSON（格式见 examples）')
    p.add_argument('--arm-id', help='记录实体机械臂身份；默认与 CAN 通道同名')
    p.add_argument('--min-corners', type=int, default=10)
    p.add_argument('--max-reprojection-px', type=float, default=1.)
    p.add_argument('--target-samples', type=int, default=30)
    p.add_argument('--max-motion-angle-error-deg', type=float, default=8., help='视觉板与机器人相对转角差的上限')
    args = p.parse_args()
    board, config = load_board(args.board_config, args.dictionary)
    total_corners = len(board.getChessboardCorners())
    if (not 4 <= args.min_corners <= total_corners or args.target_samples < 1 or args.max_reprojection_px <= 0 or
            not 0 < args.max_motion_angle_error_deg < 180):
        p.error(f'角点阈值必须为 4..{total_corners}；采样数量、重投影阈值和转角差阈值也必须有效')
    out = inside_folder(args.out)
    if out.exists():
        raise ValueError('数据目录已存在；请使用新目录，避免混合标定会话')
    camera = reader = None
    count = 0
    accepted_samples = []
    rejected_count = 0
    next_feedback_print = 0.
    try:
        if args.mode != 'intrinsics' and args.pose_source == 'can':
            reader = CanPoseReader(args.can, args.firmware, args.joint_source)
            reader.wait_ready()
            print(f'CAN {args.can}: {args.joint_source} 完整、新鲜反馈检查通过；停稳后才能保存。', flush=True)
            if args.joint_source == 'leader':
                print('当前 leader 来源是主臂角度/发给从臂的关节指令。若相机或标定板装在从臂，'
                      '请退出并用该从臂实际反馈所在通道的 --joint-source measured 重新采集；'
                      '主从共用 CAN 时仍可使用同一通道。反馈完整不能证明对应实体臂选择正确。', flush=True)
            else:
                print('当前 measured 来源是实际关节反馈；需对应装相机或标定板的实体臂。'
                      '主从控制可照常进行，采样时等待这只实体臂停稳。', flush=True)
        camera = RealSenseCamera(args.serial, args.width, args.height, args.fps, args.intrinsics, args.with_depth)
        print(f'RGB 相机 {camera.serial}: {camera.raw["width"]}x{camera.raw["height"]}@{camera.raw["fps"]} fps，'
              f'USB {camera.device_info["usb_type"]}。', flush=True)
        if args.mode != 'intrinsics' and camera.model is None:
            raise ValueError(camera.model_error)
        if camera.model is not None and args.mode != 'intrinsics':
            intrinsic_source = str(Path(args.intrinsics).resolve()) if args.intrinsics else 'RGB 工厂内参'
            print(f'RGB 内参：{intrinsic_source}；fx={camera.model["K"][0][0]:.3f}、'
                  f'fy={camera.model["K"][1][1]:.3f}，模型 {camera.model["distortion_model"]}。', flush=True)
        detector = BoardDetector(board, camera.model if args.mode != 'intrinsics' else None,
                                 args.min_corners, args.max_reprojection_px)
        out.mkdir(parents=True)
        (out / 'images').mkdir()
        (out / 'samples').mkdir()
        metadata = {'schema_version': 1, 'created_utc': utc_now(), 'synthetic': False,
                    'mode': args.mode, 'board': config, 'camera_raw': camera.raw, 'camera': camera.model,
                    'camera_device': camera.device_info,
                    'robot': None if args.mode == 'intrinsics' else {
                        'arm_id': args.arm_id or args.can, 'channel': args.can if reader else None,
                        'pose_source': args.pose_source, 'joint_source': args.joint_source if reader else None,
                        'firmware': args.firmware if reader else None, 'parent_frame': 'base', 'child_frame': 'flange'},
                    'transform_convention': 'T_parent_child maps child to parent; row-major 4x4; metres',
                    'motion_quality': {'minimum_translation_mm': 5., 'minimum_rotation_deg': 3.,
                                       'maximum_relative_rotation_disagreement_deg': args.max_motion_angle_error_deg},
                    'synchronization': 'static pose bracket; host arrival clock, not hardware-trigger synchronization'}
        write_json(out / 'metadata.json', metadata)
        print(f'ChArUco {config["squares_y"]} 行×{config["squares_x"]} 列，'
              f'方格 {config["square_length_m"] * 1000:g} mm、marker {config["marker_length_m"] * 1000:g} mm，'
              f'{config["dictionary"]}；完整板 {total_corners} 个内角点，本次至少需 {args.min_corners} 个。', flush=True)
        print(f'输出 {out}\nSPACE 采样；Q/ESC 退出。目标 {args.target_samples} 张；保持停稳。', flush=True)
        if args.mode == 'eye-in-hand':
            print('板必须固定，相机刚性安装于末端。', flush=True)
        elif args.mode == 'eye-to-hand':
            print('相机必须固定，板刚性安装于末端。', flush=True)
        while count < args.target_samples:
            image, frame_meta = camera.frame()
            reason = 'READY'
            detection = None
            try:
                detection = detector.detect(image)
                display = detector.draw(image, detection)
            except ValueError as exc:
                reason, display = str(exc), image.copy()
            panel = display.copy()
            cv2.rectangle(panel, (5, 5), (min(display.shape[1] - 5, 1190), 155 if reader else 75), (15, 15, 15), -1)
            display = cv2.addWeighted(panel, .8, display, .2, 0)
            # OpenCV's built-in font is ASCII; details are printed to the terminal on rejection.
            cv2.putText(display, f'{args.mode} {count}/{args.target_samples} SPACE=capture Q=quit',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.putText(display, 'BOARD OK' if reason == 'READY' else 'BOARD QUALITY FAIL',
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0) if reason == 'READY' else (0, 0, 255), 2)
            if reader:
                try:
                    live = reader.sample()
                    q = ' '.join(f'{v:.2f}' for v in np.rad2deg(live['joints_rad']))
                    xyz = np.array(live['T_base_flange'])[:3, 3]
                    cv2.putText(display, f'CAN {args.can}/{args.joint_source} q(deg): {q}', (15, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
                    text = f'Flange xyz(m): {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}'
                    if accepted_samples:
                        row = compare_capture(live, detection, accepted_samples[-1])
                        text += f' | vs last: {row["flange_translation_mm"]:.2f}mm {row["flange_rotation_deg"]:.2f}deg'
                    cv2.putText(display, text, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
                    if accepted_samples and row['board_rotation_deg'] is not None:
                        cv2.putText(display, f'Board rotation: {row["board_rotation_deg"]:.2f}deg | '
                                    f'rotation disagreement: {row["rotation_disagreement_deg"]:.2f}deg', (15, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55,
                                    (0, 0, 255) if row['rotation_disagreement_deg'] > args.max_motion_angle_error_deg else (0, 255, 0), 1)
                except ValueError as exc:
                    cv2.putText(display, 'CAN FEEDBACK UNAVAILABLE', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                    if time.monotonic() >= next_feedback_print:
                        print(f'CAN 预览：{exc}', flush=True)
                        next_feedback_print = time.monotonic() + 3.
            cv2.imshow('D435i ChArUco calibration', display)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty('D435i ChArUco calibration', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key != 32:
                continue
            try:
                packet = None
                if reader:
                    packets = reader.stationary_window(.5)
                    packets.append(reader.sample())
                    image, frame_meta = camera.frame(flush=True)
                    packets.append(reader.sample())
                    packets += reader.stationary_window(.2)
                    require_stationary(packets)
                    # Bracketed, static samples use nearest host reception time. Different
                    # device clocks are recorded, never subtracted from each other.
                    packet = min(packets, key=lambda s: abs(s['host_monotonic_s'] - frame_meta['host_monotonic_s'])).copy()
                    packet['stability_checked'] = True
                    packet['static_window_s'] = packets[-1]['host_monotonic_s'] - packets[0]['host_monotonic_s']
                    packet['host_pair_offset_s'] = abs(packet['host_monotonic_s'] - frame_meta['host_monotonic_s'])
                    if packet['host_pair_offset_s'] > .1:
                        raise ValueError('图像/位姿主机接收时间相差过大')
                elif args.mode != 'intrinsics':
                    packet = manual_pose(args.pose_file)
                    image, frame_meta = camera.frame(flush=True)
                else:
                    image, frame_meta = camera.frame(flush=True)
                detection = detector.detect(image)
                if packet:
                    motion_comparisons = check_capture(packet, detection, accepted_samples, args.max_motion_angle_error_deg)
                name = f'{count:04d}'
                image_path = out / 'images' / f'{name}.png'
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError('图像写入失败')
                sample = {'id': name, 'utc': utc_now(), 'image': f'images/{name}.png',
                          'frame': frame_meta, 'robot': packet, **detection}
                if packet:
                    sample['motion_comparisons'] = motion_comparisons
                write_json(out / 'samples' / f'{name}.json', sample)
                count += 1
                if packet:
                    accepted_samples.append(sample)
                print(f'已采样 {count}/{args.target_samples}, corners={len(detection["corner_ids"])}, '
                      f'rms={detection.get("reprojection_rms_px", "intrinsics-only")}', flush=True)
            except MotionCaptureError as exc:
                stem = f'{rejected_count:04d}'
                folder = out / 'rejections'
                folder.mkdir(exist_ok=True)
                if not cv2.imwrite(str(folder / f'{stem}.png'), image):
                    raise RuntimeError('诊断图像写入失败')
                write_json(folder / f'{stem}.json', {'utc': utc_now(), 'reason': str(exc), 'diagnostic': exc.report,
                    'image': f'rejections/{stem}.png', 'robot': packet, 'frame': frame_meta, 'detection': detection})
                rejected_count += 1
                print(f'本次未保存为标定样本：{exc}\n诊断记录: {folder / (stem + ".json")}', flush=True)
            except ValueError as exc:
                print(f'本次未保存：{exc}', flush=True)
        print(f'结束：保存 {count} 个样本到 {out}', flush=True)
    finally:
        if reader:
            reader.close()
        if camera:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f'采集失败：{exc}')
