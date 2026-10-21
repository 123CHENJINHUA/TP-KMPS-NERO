"""Vision entry points. These commands never send robot movement instructions."""
from .settings import DEFAULT_CONFIG, apply_settings
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture', help='D435i 静态 RGB-D + 从臂 measured；不运动')
    capture.add_argument('--out', required=True)
    capture.add_argument('--serial', help='覆盖相机配置中的序列号')
    capture.add_argument('--width', type=int)
    capture.add_argument('--height', type=int)
    capture.add_argument('--fps', type=int)
    capture.add_argument('--frames', type=int)
    capture_intrinsics = capture.add_mutually_exclusive_group()
    capture_intrinsics.add_argument('--intrinsics', help='覆盖相机配置中的内参 JSON 路径')
    capture_intrinsics.add_argument('--factory-intrinsics', action='store_true', help='本次使用出厂内参，覆盖 JSON 中的 intrinsics')
    capture.add_argument('--config')
    capture.add_argument('--can')
    capture.add_argument('--no-robot', action='store_true', help='只拍照；眼在手上转换时会拒绝缺少机器人反馈的数据')
    estimate = commands.add_parser('estimate', help='自动分割、CAD 配准、几何质检和手眼坐标转换')
    estimate.add_argument('--capture', required=True)
    estimate.add_argument('--out', required=True)
    estimate.add_argument('--object')
    hand_eye = estimate.add_mutually_exclusive_group()
    hand_eye.add_argument('--hand-eye', help='覆盖相机配置中的手眼 result.json 路径')
    hand_eye.add_argument('--camera-only', action='store_true', help='本次只输出相机位姿，不使用配置中的手眼结果')
    estimate.add_argument('--mask-method', choices=['box', 'sam3', 'hsv', 'file'])
    estimate.add_argument('--box', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='框选模式的归一化 XYXY；省略则鼠标框选')
    estimate.add_argument('--mask-file')
    estimate.add_argument('--prompt')
    estimate.add_argument('--instance', type=int)
    estimate.add_argument('--sam3-python')
    estimate.add_argument('--sam3-checkpoint')
    estimate.add_argument('--sam3-source')
    estimate.add_argument('--pose-python')
    estimate.add_argument('--foundationpose-root')
    estimate.add_argument('--iterations', type=int)
    estimate.add_argument('--diagnostic-poses', help='导入其他算法的 N×4×4 位姿 NPZ；强制诊断标记，不能真机执行')
    tracking = commands.add_parser('track', help='独立 RGB-D 连续跟踪预览/测量；不读 CAN、不使用手眼、不运动')
    tracking.add_argument('--out', required=True)
    tracking.add_argument('--object', help='默认 objects/keel.json；Meshy 模型用导入后的 object.json')
    tracking.add_argument('--serial', help='覆盖相机配置中的序列号')
    tracking.add_argument('--width', type=int)
    tracking.add_argument('--height', type=int)
    tracking.add_argument('--fps', type=int)
    tracking_intrinsics = tracking.add_mutually_exclusive_group()
    tracking_intrinsics.add_argument('--intrinsics', help='覆盖相机配置中的内参 JSON 路径；回放沿用采集记录')
    tracking_intrinsics.add_argument('--factory-intrinsics', action='store_true', help='本次使用出厂内参，覆盖 JSON 中的 intrinsics')
    tracking.add_argument('--mask-method', choices=['box', 'sam3', 'hsv'])
    tracking.add_argument('--box', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='初始归一化 XYXY；框选模式省略则弹出鼠标框选')
    tracking.add_argument('--prompt')
    tracking.add_argument('--instance', type=int)
    tracking.add_argument('--sam3-python')
    tracking.add_argument('--sam3-checkpoint')
    tracking.add_argument('--sam3-source')
    tracking.add_argument('--pose-python')
    tracking.add_argument('--foundationpose-root')
    tracking.add_argument('--register-iterations', type=int, default=5)
    tracking.add_argument('--track-iterations', type=int, default=2)
    tracking.add_argument('--max-width', type=int, default=640, help='配准后的 RGB/深度一起缩小，并同步缩放 K')
    tracking.add_argument('--max-step-mm', type=float, default=80.)
    tracking.add_argument('--max-step-deg', type=float, default=45.)
    tracking.add_argument('--max-gap-s', type=float, default=10.)
    tracking.add_argument('--measure-samples', type=int, default=10)
    tracking.add_argument('--measure-static', action='store_true', help='启动即采一个静止统计窗口，要求相机/物体保持不动')
    tracking.add_argument('--expected-distance-mm', type=float, help='A/B 纯平移的尺测距离；只有相机固定时可比较')
    tracking.add_argument('--replay', help='回放 capture.sh 保存的 RGB-D 目录；不打开相机或 CAN')
    tracking.add_argument('--headless', action='store_true', help='不创建窗口，保存 latest.png/轨迹和报告')
    tracking.add_argument('--seconds', type=float, default=0., help='实时模式的最长秒数，含模型加载；0 为不限')
    tracking.add_argument('--max-frames', type=int, default=0, help='最多处理多少帧；0 为不限')
    tracking.add_argument('--save-every', type=int, default=30, help='每 N 个处理帧保存图像/RGB-D 快照；0 仅手动 P')
    bind = commands.add_parser('bind-grasp', help='同一物体摆放下示教一次 object<-TCP 抓取关系')
    bind.add_argument('--observation', required=True)
    bind.add_argument('--out', required=True)
    bind.add_argument('--config')
    bind.add_argument('--can')
    bind.add_argument('--demo', help='用同一摆放下示教的 pick 参考 TCP；省略则实时只读记录')
    bind.add_argument('--update-config', action='store_true', help='把本次真实抓取绑定参数直接写入相机 JSON 的 grasp 字段')
    frames = commands.add_parser('frames', help='新物体位姿生成现有 TP-KMP 可读任务参考系')
    frames.add_argument('--observation', required=True)
    frames.add_argument('--grasp', help='覆盖相机配置中的抓取绑定 JSON 路径')
    frames.add_argument('--template-frames', required=True)
    frames.add_argument('--out', required=True)
    frames.add_argument('--config')
    for entry in (capture, estimate, tracking, frames, bind):
        entry.add_argument('--camera-config', default=str(DEFAULT_CONFIG), help='相机/视觉 JSON；默认 vision/camera_config.json')
    doctor = commands.add_parser('doctor', help='环境/权重/CUDA 诊断；不打开相机和 CAN')
    doctor.add_argument('--pose-python')
    doctor.add_argument('--sam3-python')
    doctor.add_argument('--foundationpose-root')
    doctor.add_argument('--sam3-checkpoint')
    doctor.add_argument('--out')
    commands.add_parser('self-test', help='纯离线单元测试，不加载 GPU 网络/相机/CAN')
    commands.add_parser('tracking-self-test', help='连续跟踪状态、重定位和测量的纯离线测试')
    smoke = commands.add_parser('smoke', help='生成明确标记的合成 RGB-D 并验证视觉到规划接口')
    smoke.add_argument('--out', required=True)
    gpu = commands.add_parser('gpu-check', help='真实 GPU 网络的合成新位置联调，不访问相机或 CAN')
    gpu.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command in ('capture', 'estimate', 'track', 'frames'):
        try:
            apply_settings(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.command == 'capture':
        from .capture import capture
        capture(args)
    elif args.command == 'estimate':
        if args.iterations < 1:
            parser.error('iterations 必须为正整数')
        if args.box is not None and args.mask_method != 'box':
            parser.error('--box 需要 --mask-method box')
        if args.mask_method == 'box' and args.instance is not None:
            parser.error('框选模式不使用 instance；请直接框住一个物体')
        from .estimate import estimate
        estimate(args)
    elif args.command == 'track':
        import math
        positive = (args.max_step_mm, args.max_step_deg, args.max_gap_s)
        if (any(not math.isfinite(v) or v <= 0 for v in positive)
                or not math.isfinite(args.seconds) or args.seconds < 0
                or args.max_frames < 0 or args.save_every < 0 or args.measure_samples < 3
                or args.register_iterations < 1 or args.track_iterations < 1 or args.max_width < 160
                or (not args.replay and min(args.width, args.height, args.fps) <= 0)
                or (args.instance is not None and args.instance < 0)
                or (args.expected_distance_mm is not None and
                    (not math.isfinite(args.expected_distance_mm) or args.expected_distance_mm <= 0))):
            parser.error('跟踪迭代/尺度/门限需要正数；静止样本数至少 3；时长/帧数/快照间隔不能为负')
        if args.box is not None and args.mask_method != 'box':
            parser.error('--box 需要 --mask-method box')
        if args.mask_method == 'box' and args.instance is not None:
            parser.error('框选模式不使用 instance；请直接框住一个物体')
        if args.mask_method == 'box' and args.headless and args.box is None:
            parser.error('无窗口的框选模式需要 --box x0 y0 x1 y1（归一化 0–1）')
        from .tracking import track
        track(args)
    elif args.command == 'bind-grasp':
        from .bindings import bind
        result = bind(args)
        if args.update_config and result is not None:
            from .settings import update_grasp_parameters
            from ..data import config
            update_grasp_parameters(args.camera_config, result, cfg=config(args.config, args.can))
            print(f'抓取参数已写入：{args.camera_config} 的 grasp 字段')
    elif args.command == 'frames':
        from .bindings import export_frames
        export_frames(args)
    elif args.command == 'doctor':
        from .doctor import doctor
        doctor(args)
    elif args.command == 'self-test':
        from .self_test import run
        run()
    elif args.command == 'tracking-self-test':
        from .tracking_test import run
        run()
    elif args.command == 'smoke':
        from .smoke import smoke
        smoke(args)
    elif args.command == 'gpu-check':
        from .gpu_check import gpu_check
        gpu_check(args)


if __name__ == '__main__':
    main()
