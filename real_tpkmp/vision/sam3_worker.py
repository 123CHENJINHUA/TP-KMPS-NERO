"""Standalone worker: deliberately avoids the TP-KMP/vendor NumPy import paths."""
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--instance', type=int)
    args = p.parse_args()
    if Path(args.source).is_dir():
        sys.path.insert(0, args.source)
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from box_prompt import checked_box, select_box_candidate
    if not torch.cuda.is_available():
        raise RuntimeError('SAM3 worker 看不到 CUDA；检查当前主机驱动和所选 Python 环境')
    with np.load(args.input, allow_pickle=False) as data:
        frames = data['rgb']
        boxes = data['boxes_xyxy'] if 'boxes_xyxy' in data else None
    if boxes is not None and boxes.shape != (len(frames), 4):
        raise ValueError('需要每帧一个归一化 XYXY 框')
    if boxes is None and not args.prompt:
        raise ValueError('需要文本描述或框提示')
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False, device='cuda',
                                  enable_inst_interactivity=boxes is not None)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=args.threshold)
    masks, scores = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, frame in enumerate(frames):
            state = processor.set_image(Image.fromarray(frame))
            if boxes is not None:
                box = checked_box(boxes[i])
                h, w = frame.shape[:2]
                all_masks, values, _ = model.predict_inst(state, box=box*np.array([w, h, w, h]),
                                                        multimask_output=False, normalize_coords=True)
                all_masks = np.asarray(all_masks, bool).reshape(-1, h, w)
                values = np.asarray(values, float).reshape(-1)
                selected = select_box_candidate(all_masks, values, box, args.threshold)
                masks.append(all_masks[selected])
                scores.append(float(values[selected]))
                continue
            result = processor.set_text_prompt(prompt=args.prompt, state=state)
            values = result['scores'].detach().float().cpu().numpy().reshape(-1)
            all_masks = result['masks'].detach().float().cpu().numpy().reshape(-1, *frame.shape[:2]) > .5
            candidates = np.flatnonzero(values >= args.threshold)
            candidates = candidates[np.argsort(-values[candidates])]
            if len(candidates) == 0:
                raise ValueError(f'SAM3 没有找到 {args.prompt!r}')
            if args.instance is None and len(candidates) != 1:
                raise ValueError(f'SAM3 找到 {len(candidates)} 个候选；请限定场景或指定 --instance')
            index = 0 if args.instance is None else args.instance
            if not 0 <= index < len(candidates):
                raise ValueError('instance 超出 SAM3 候选范围')
            selected = candidates[index]
            masks.append(all_masks[selected])
            scores.append(float(values[selected]))
    with Path(args.output).open('xb') as stream:
        np.savez_compressed(stream, masks=np.array(masks), scores=np.array(scores))


if __name__ == '__main__':
    main()
