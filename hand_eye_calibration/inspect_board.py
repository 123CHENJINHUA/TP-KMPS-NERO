"""Suggest dictionaries for an existing board; shared codewords remain ambiguous."""
from bootstrap import HERE, read_json, write_json, inside_folder, utc_now
import argparse
from datetime import datetime
import cv2
import numpy as np
from board_diagnostics import diagnose_layout, draw_layout


def inspect(image, config):
    candidates = []
    for name in sorted(n for n in dir(cv2.aruco) if n.startswith('DICT_') and isinstance(getattr(cv2.aruco, n), int)):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
        if ids is None:
            continue
        args = ((config['squares_x'], config['squares_y']), config['square_length_m'], config['marker_length_m'], dictionary)
        error = None
        try:
            custom = config.get('marker_ids')
            board = cv2.aruco.CharucoBoard(*args) if custom is None else cv2.aruco.CharucoBoard(*args, np.array(custom, np.int32))
            board.setLegacyPattern(config.get('legacy_pattern', False))
            cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image, markerCorners=corners, markerIds=ids)
            count = 0 if ci is None else len(ci)
        except cv2.error as exc:
            count = 0
            error = str(exc)
        candidates.append({'dictionary': name, 'markers': len(ids), 'marker_ids': sorted(ids.ravel().tolist()),
                           'charuco_corners_matching_config': count, 'opencv_error': error})
    candidates.sort(key=lambda c: (c['charuco_corners_matching_config'], c['markers']), reverse=True)
    return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', help='现有板的清晰照片；不提供则打开 D435i 预览')
    p.add_argument('--serial')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--board-config', default=str(HERE / 'board_config.json'))
    p.add_argument('--out', help='可选 JSON 报告')
    p.add_argument('--diagnostic-dir', help='自动保存原图、标注图、布局比较及建议配置；默认时间戳目录')
    args = p.parse_args()
    config = read_json(args.board_config)
    folder = inside_folder(args.diagnostic_dir or HERE / 'diagnostics' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if folder.exists():
        raise ValueError('诊断目录已存在；请换一个 --diagnostic-dir')
    frame = None
    camera_raw = camera_model = None
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError('照片打不开')
    else:
        from realsense_camera import RealSenseCamera
        camera = RealSenseCamera(serial=args.serial, width=args.width, height=args.height, fps=args.fps)
        camera_raw, camera_model = camera.raw, camera.model
        image = None
        print('让板完整入镜，按空格检查；Q 退出。')
        try:
            while True:
                current, frame = camera.frame()
                cv2.imshow('SPACE: inspect board, Q: quit', current)
                key = cv2.waitKey(1) & 0xff
                if key == 32:
                    image = current
                    break
                if key in (27, ord('q')):
                    return
        finally:
            camera.close()
            cv2.destroyAllWindows()
    candidates = inspect(image, config)
    import json
    print(json.dumps(candidates, indent=2, ensure_ascii=False))
    print('同一系列的标准字典可能共用前若干编号/码字；不能唯一识别时需查板的生成配置。')
    best_dictionary = config.get('dictionary')
    top = max((c['markers'] for c in candidates), default=0)
    current = next((c for c in candidates if c['dictionary'] == best_dictionary), None)
    if current is None or current['markers'] != top:
        best_dictionary = candidates[0]['dictionary'] if candidates else None
    layouts, suggested, corners, ids = [], None, [], None
    if best_dictionary:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, best_dictionary))
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
        layout_config = dict(config, dictionary=best_dictionary)
        if ids is not None:
            layouts, suggested = diagnose_layout(image, layout_config, dictionary, corners, ids)
    folder.mkdir(parents=True)
    if not cv2.imwrite(str(folder / 'raw.png'), image):
        raise RuntimeError('原图保存失败')
    best = layouts[0] if layouts else None
    annotated = draw_layout(image, corners, ids, best)
    if not cv2.imwrite(str(folder / 'annotated.png'), annotated):
        raise RuntimeError('标注图保存失败')
    report = {'created_utc': utc_now(), 'opencv_version': cv2.__version__, 'configured_board': config,
              'camera_raw': camera_raw, 'camera': camera_model, 'frame': frame,
              'candidates': candidates, 'layout_dictionary': best_dictionary, 'layouts': layouts,
              'suggested_board_config': suggested, 'unique_dictionary_confirmed': False}
    write_json(folder / 'report.json', report)
    print('\n布局比较（先按全部 marker 的几何一致性排序）：')
    for row in layouts[:8]:
        rms = row['marker_geometry']['rms_px']
        print(f'  {row["layout"]}: ChArUco={row["charuco_corners"]}, '
              f'marker几何RMS={"n/a" if rms is None else f"{rms:.2f}px"}, '
              f'完整板候选={row["plausible_full_board"]}')
        if row['opencv_error']:
            print('    OpenCV error: ' + row['opencv_error'])
    if suggested:
        write_json(folder / 'suggested_board_config.json', suggested)
        print(f'该图片有一个符合几何与角点检查的布局；建议配置: {folder / "suggested_board_config.json"}')
        print(f'确认标注的棋盘角点位置正确，并实测方格为 {suggested["square_length_m"] * 1000:g} mm、'
              f'marker 为 {suggested["marker_length_m"] * 1000:g} mm 后，可用 --board-config 指定它。')
    else:
        print('没有唯一符合检查的完整板布局；请查看原图/标注图、板生成配置及 report.json。')
    print(f'诊断文件: {folder}\n当前 board_config.json 未改动。')
    if args.out:
        write_json(args.out, report)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
