"""Meshy 单图/多图生成及模型导入。generate/submit 会上传照片并消耗账户积分。"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='1–4 张本地照片创建云端任务并下载；--dry-run 仅本地检查')
    generate.add_argument('--images', nargs='+', required=True)
    generate.add_argument('--out', required=True, help='新的视觉目录内任务文件夹')
    generate.add_argument('--ai-model', choices=['meshy-7.1', 'latest'], default='meshy-7.1')
    generate.add_argument('--geometry-resolution', choices=['standard', '2k'], default='standard')
    generate.add_argument('--texture', action='store_true', help='额外生成纹理；可能增加积分和时间')
    generate.add_argument('--polycount', type=int, help='可选 100–300000，启用重拓扑；省略保留原始几何')
    generate.add_argument('--dry-run', action='store_true', help='不联网、不要求 API key；保存照片副本和参数')
    submit = commands.add_parser('submit', help='提交本地 dry-run 目录；会上传照片并消耗积分')
    submit.add_argument('--run', required=True)
    resume = commands.add_parser('resume', help='继续查询/下载已有任务；不会创建付费任务')
    resume.add_argument('--run', required=True)
    resume.add_argument('--task-id', help='仅用于提交响应丢失后，从 Meshy 网页找回的任务编号')
    for command in (generate, submit, resume):
        command.add_argument('--wait-seconds', type=float, default=1200., help='本地总等待秒数，不等于取消云端任务')
        command.add_argument('--poll-seconds', type=float, default=10.)
    model = commands.add_parser('import', help='离线导入网页/API 的 GLB/STL 等模型，实测尺寸均匀缩放')
    model.add_argument('--mesh', required=True)
    model.add_argument('--object-id', required=True, help='使用新名字，例如 keel_meshy，不能覆盖原 CAD')
    model.add_argument('--measured-extents-mm', nargs=3, type=float, required=True, metavar=('L', 'W', 'H'))
    model.add_argument('--rotate-xyz-deg', nargs=3, type=float, default=[0., 0., 0.], metavar=('RX', 'RY', 'RZ'))
    model.add_argument('--prompt', required=True, help='SAM3 的目标描述，例如 blue metal bracket')
    model.add_argument('--out', help='默认 vision/models/<object-id>，必须是新目录')
    model.add_argument('--pose-python', help='只借用该环境的 trimesh，不加载 GPU 网络')
    commands.add_parser('self-test', help='本地 mock API、尺寸和导入测试；不联网或连接硬件')
    args = parser.parse_args()
    if args.command in ('generate', 'submit', 'resume'):
        import math
        if (not math.isfinite(args.wait_seconds) or args.wait_seconds < 0
                or not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60):
            parser.error('wait-seconds 必须非负，poll-seconds 需要在 1–60 秒之间')
    if args.command == 'generate':
        if args.polycount is not None and not 100 <= args.polycount <= 300000:
            parser.error('polycount 需要在 100–300000 之间')
        from .api import generate
        generate(args)
    elif args.command == 'submit':
        from .api import read_job, submit
        directory, manifest = read_job(args.run)
        submit(directory, manifest, args)
    elif args.command == 'resume':
        from .api import resume
        resume(args)
    elif args.command == 'import':
        from .models import import_model
        import_model(args)
    elif args.command == 'self-test':
        from .self_test import run
        run()


if __name__ == '__main__':
    main()
