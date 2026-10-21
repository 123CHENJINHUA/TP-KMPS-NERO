"""Persistent isolated GPU worker. JSON control on stdout; all library output in log."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


class Segmenter:
    def __init__(self, cfg):
        sys.path.insert(0, cfg['sam3_source'])
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not torch.cuda.is_available():
            raise RuntimeError('SAM3 看不到 CUDA；请在主机终端检查 nvidia-smi')
        model = build_sam3_image_model(checkpoint_path=cfg['sam3_checkpoint'], load_from_HF=False, device='cuda',
                                      enable_inst_interactivity=cfg.get('box_prompt_mode', False))
        self.processor = Sam3Processor(model, device='cuda', confidence_threshold=cfg['threshold'])
        self.cfg = cfg

    def process(self, req):
        import numpy as np
        import torch
        from PIL import Image
        from box_prompt import checked_box, select_box_candidate
        with np.load(req['input'], allow_pickle=False) as data:
            rgb = data['rgb']
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            box = req.get('box_xyxy')
            if box is not None:
                box = checked_box(box)
                h, w = rgb.shape[:2]
                all_masks, scores, _ = self.processor.model.predict_inst(
                    state, box=box*np.array([w, h, w, h]), multimask_output=False, normalize_coords=True)
                all_masks = np.asarray(all_masks, bool).reshape(-1, h, w)
                scores = np.asarray(scores, float).reshape(-1)
                selected = select_box_candidate(all_masks, scores, box, self.cfg['threshold'])
            else:
                result = self.processor.set_text_prompt(prompt=self.cfg['prompt'], state=state)
                scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                all_masks = result['masks'].detach().float().cpu().numpy().reshape(-1, *rgb.shape[:2]) > .5
                selected = self.select_text_candidate(scores)
                del result
            mask = all_masks[selected]
            score = float(scores[selected])
        np.save(req['output'], mask, allow_pickle=False)
        del state, all_masks
        torch.cuda.empty_cache()  # Let the other persistent worker use temporary buffers.
        return dict(score=score, box_xyxy=box.tolist() if box is not None else None)

    def select_text_candidate(self, scores):
        import numpy as np
        candidates = np.flatnonzero(scores >= self.cfg['threshold'])
        candidates = candidates[np.argsort(-scores[candidates])]
        if not len(candidates):
            raise ValueError('SAM3 没找到目标')
        instance = self.cfg['instance']
        if instance is None and len(candidates) != 1:
            raise ValueError(f'SAM3 有 {len(candidates)} 个候选；目标不唯一')
        index = 0 if instance is None else instance
        if not 0 <= index < len(candidates):
            raise ValueError('instance 超出当前候选范围')
        return candidates[index]


class PoseTracker:
    def __init__(self, cfg):
        root = Path(cfg['foundationpose_root']).resolve()
        if not (root/'estimater.py').is_file() and (root/'FoundationPose/estimater.py').is_file():
            root /= 'FoundationPose'
        if not (root/'estimater.py').is_file():
            raise FileNotFoundError('找不到 FoundationPose estimater.py')
        sys.path.insert(0, str(root))
        os.chdir(root)
        import numpy as np
        import torch
        import trimesh
        if not torch.cuda.is_available():
            raise RuntimeError('FoundationPose 看不到 CUDA；请在主机终端检查 nvidia-smi')
        includes = list((Path(torch.__file__).resolve().parents[1]/'nvidia').glob('*/include'))
        os.environ['CPATH'] = os.pathsep.join(map(str, includes))+os.pathsep+os.environ.get('CPATH', '')
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}')
        import warp
        warp.config.kernel_cache_dir = str(Path(__file__).resolve().parent/'.cache/warp')
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr
        from Utils import trimesh_add_pure_colored_texture
        mesh = trimesh.load(cfg['mesh_path'], force='mesh')
        mesh.apply_scale(cfg['mesh_scale_to_m'])
        if cfg.get('mesh_rgb') is not None:
            mesh = trimesh_add_pure_colored_texture(mesh, color=np.asarray(cfg['mesh_rgb']), resolution=10)
        self.estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                       scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                                       glctx=dr.RasterizeCudaContext(), debug=0, debug_dir=cfg['debug_dir'])
        self.cfg = cfg

    def process(self, req):
        import numpy as np
        import torch
        with np.load(req['input'], allow_pickle=False) as data:
            rgb, depth, K = data['rgb'], data['depth_m'], data['K']
        registration = bool(req['reset'] or self.estimator.pose_last is None)
        if registration:
            mask = np.load(req['mask'], allow_pickle=False).astype(bool)
            pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                           iteration=self.cfg['register_iterations'])
        else:
            pose = self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=self.cfg['track_iterations'])
        pose = np.asarray(pose, float).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise RuntimeError('跟踪网络返回非有限位姿')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return dict(T_color_object=pose.tolist(), registered=registration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['pose', 'sam3'], required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    control = sys.stdout
    def reply(value):
        print(json.dumps(value, allow_nan=False), file=control, flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
            model = PoseTracker(cfg) if args.role == 'pose' else Segmenter(cfg)
        except BaseException as exc:
            traceback.print_exc()
            reply(dict(ok=False, fatal=True, error=f'{type(exc).__name__}: {exc}'))
            return
        reply(dict(ok=True, ready=True))
        for line in sys.stdin:
            start = time.monotonic()
            try:
                req = json.loads(line)
                result = model.process(req)
                reply(dict(ok=True, elapsed_ms=(time.monotonic()-start)*1000, **result))
            except ValueError as exc:
                reply(dict(ok=False, fatal=False, error=str(exc)))
            except BaseException as exc:
                traceback.print_exc()
                reply(dict(ok=False, fatal=True, error=f'{type(exc).__name__}: {exc}'))
                return


if __name__ == '__main__':
    main()
