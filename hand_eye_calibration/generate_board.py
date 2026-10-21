"""Generate a known ChArUco board, including a physically sized A4 PDF."""
from bootstrap import HERE, inside_folder, write_json
import argparse
import cv2
import numpy as np
from vision import load_board


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dictionary', required=True)
    p.add_argument('--board-config', default=str(HERE / 'board_config.json'))
    p.add_argument('--out', help='默认按实际行列命名；不覆盖已有目录')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()
    board, config = load_board(args.board_config, args.dictionary)
    out = inside_folder(args.out or HERE / 'boards' / f'charuco_rows{config["squares_y"]}_cols{config["squares_x"]}')
    if out.exists():
        raise ValueError('输出目录已存在；请更换 --out')
    if not 100 <= args.dpi <= 1200:
        raise ValueError('dpi 必须在 100..1200')
    width_mm = config['squares_x'] * config['square_length_m'] * 1000
    height_mm = config['squares_y'] * config['square_length_m'] * 1000
    paper_width, paper_height = 210, 297
    if width_mm > 200 or height_mm > 287:
        paper_width, paper_height = 297, 210
    if width_mm > paper_width - 10 or height_mm > paper_height - 10:
        raise ValueError('此板不适合 A4 的至少 5mm 边距；请调整 PDF 纸张尺寸')
    px_square = round(args.dpi * config['square_length_m'] / .0254)
    effective_dpi = px_square * .0254 / config['square_length_m']
    image = board.generateImage((px_square * config['squares_x'], px_square * config['squares_y']), marginSize=0, borderBits=1)
    out.mkdir(parents=True)
    from PIL import Image
    Image.fromarray(image).save(out / 'board.png', dpi=(effective_dpi, effective_dpi))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(paper_width / 25.4, paper_height / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    x0, y0 = (paper_width - width_mm) / 2, (paper_height - height_mm) / 2
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=(x0, x0 + width_mm, y0, y0 + height_mm), origin='upper', interpolation='none')
    ax.set(xlim=(0, paper_width), ylim=(0, paper_height))
    ax.set_axis_off()
    fig.savefig(out / 'board_A4_100percent.pdf')  # no bbox_inches=tight: preserve physical page
    plt.close(fig)
    write_json(out / 'board_config.json', config)
    write_json(out / 'print_dimensions.json', {'paper_mm': [paper_width, paper_height], 'board_mm': [width_mm, height_mm],
               'square_mm': config['square_length_m'] * 1000, 'marker_mm': config['marker_length_m'] * 1000,
               'png_dpi': effective_dpi, 'png_shape': list(image.shape), 'print_scale_percent': 100})
    print(f'保存 {out}；PDF 按实际大小 100% 打印，禁止适合页面。'
          f'打印后实测方格必须为 {config["square_length_m"] * 1000:g} mm，'
          f'marker 为 {config["marker_length_m"] * 1000:g} mm。')


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc))
