"""Automatic blue-object mask or isolated SAM3 text inference; reject ambiguity."""
from .common import ROOT, output_path
import os
import subprocess
import sys
from pathlib import Path
import numpy as np


def hsv_masks(rgb, cfg, instance=None):
    import cv2
    if 'hsv_lower' not in cfg or 'hsv_upper' not in cfg:
        raise ValueError('该物体未配置 HSV 颜色范围；请使用默认 SAM3，或在物体 JSON 中设置真实颜色范围')
    masks = []
    for frame in rgb:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        binary = cv2.inRange(hsv, np.uint8(cfg['hsv_lower']), np.uint8(cfg['hsv_upper']))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary)
        candidates = [i for i in range(1, count) if statistics[i, cv2.CC_STAT_AREA] >= cfg['min_mask_pixels']]
        candidates.sort(key=lambda i: (-statistics[i, cv2.CC_STAT_AREA], statistics[i, cv2.CC_STAT_LEFT]))
        if not candidates:
            raise ValueError('HSV 没有检测到目标；只有实物为蓝色时才使用该模式')
        if instance is None and len(candidates) != 1:
            raise ValueError(f'检测到 {len(candidates)} 个蓝色候选；请限定场景或明确 --instance 编号')
        selected = 0 if instance is None else instance
        if not 0 <= selected < len(candidates):
            raise ValueError('instance 超出候选范围')
        masks.append(labels == candidates[selected])
    return np.array(masks), dict(method='hsv', automatic=True, instance=instance)


def get_masks(rgb, cfg, output, method='sam3', python=None, checkpoint=None, source=None, prompt=None, instance=None, mask_file=None, box=None):
    import cv2
    if method == 'hsv':
        return hsv_masks(rgb, cfg, instance)
    if method == 'file':
        if not mask_file:
            raise ValueError('--mask-method file 需要 --mask-file')
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != rgb.shape[1:3]:
            raise ValueError('mask 文件必须匹配已去畸变的 RGB 分辨率；不能使用原始畸变图的 mask')
        return np.repeat((mask > 0)[None], len(rgb), axis=0), dict(method='file', automatic=False)
    if method not in ('sam3', 'box'):
        raise ValueError('mask-method 需要 box、sam3、hsv 或 file')
    selected_box = None
    if method == 'box':
        from .box_prompt import checked_box, select_box
        selected_box = checked_box(box).tolist() if box is not None else select_box(rgb[0])
        if selected_box is None:
            raise ValueError('已取消框选，没有执行分割')
        if instance is not None:
            raise ValueError('框选模式不使用 instance 排名，请直接框住一个目标')
    request = output_path(output/'segmentation_request.npz')
    with request.open('xb') as stream:
        extra = dict(boxes_xyxy=np.repeat(np.array(selected_box)[None], len(rgb), axis=0)) if selected_box is not None else {}
        np.savez_compressed(stream, rgb=rgb, **extra)
    destination = output/'sam3_masks.npz'
    python = python or os.environ.get('SAM3_PYTHON', str(ROOT/'envs/sam3/bin/python'))
    checkpoint = Path(checkpoint or ROOT/'weights/sam3.pt').resolve()
    if not Path(python).is_file() or not checkpoint.is_file():
        raise ValueError('SAM3 环境/权重未就绪；运行 setup.sh 和 doctor.sh，或对蓝色实物使用 --mask-method hsv')
    command = [python, '-B', str(ROOT/'sam3_worker.py'), '--input', str(request), '--output', str(destination),
               '--checkpoint', str(checkpoint), '--source', str(Path(source or ROOT/'third_party/sam3').resolve()),
               '--threshold', str(cfg['min_segmentation_score'])]
    if selected_box is None:
        command += ['--prompt', prompt or cfg['text_prompt']]
    if instance is not None:
        command += ['--instance', str(instance)]
    print(f'SAM3 分割 {len(rgb)} 帧，'+(f'框提示：{selected_box}' if selected_box else f'文字提示：{prompt or cfg["text_prompt"]}'), flush=True)
    with (output/'sam3.log').open('x') as log:
        try:
            subprocess.run(command, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'SAM3 分割失败，详见 {output / "sam3.log"}') from exc
    with np.load(destination, allow_pickle=False) as result:
        masks = np.asarray(result['masks'], bool)
        scores = result['scores'].tolist()
    if masks.shape != rgb.shape[:3]:
        raise ValueError('SAM3 输出 mask 分辨率与 RGB 不匹配')
    return masks, dict(method=method, automatic=method != 'box',
                       prompt=None if selected_box else prompt or cfg['text_prompt'],
                       box_xyxy=selected_box, scores=scores, instance=instance)
