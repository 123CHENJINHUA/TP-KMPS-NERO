"""On-demand visual pick targets for passive six-stage demonstration collection."""
from .bootstrap import read_json, write_json, digest, new_directory, utc_now
from .data import physical_signature, read_frames, frame_document
from .kinematics import pose_matrix
from .vision.common import ROOT, object_config
from .vision.bindings import grasp_target, frame_reasons
from .vision.settings import read_settings, materialize_settings
from argparse import Namespace
from pathlib import Path
import os
import time
import numpy as np


def load_settings(path, cfg):
    """Validate static inputs before opening CAN; relative paths use the project root."""
    settings = read_settings(path)
    for key in ('hand_eye', 'grasp'):
        if not settings.get(key):
            raise ValueError(f'请在 {path} 的 {key} 字段填写结果参数；先完成手眼标定和 bind-grasp')
    if isinstance(settings['hand_eye'], dict) and settings['hand_eye'].get('passed') is not True:
        raise ValueError('手眼参数尚未通过验证；填写真实矩阵和标定信息，验证后再设 passed=true')
    settings = materialize_settings(settings, cfg=cfg)
    calibration = read_json(settings['hand_eye'])
    if calibration.get('passed') is not True or calibration.get('synthetic'):
        raise ValueError('视觉采集需要通过验证的真实手眼标定结果')
    transform_name = {'eye-in-hand': 'T_flange_color', 'eye-to-hand': 'T_base_color'}.get(calibration.get('mode'))
    if (calibration.get('schema_version') != 1 or transform_name is None
            or calibration.get('transform_name') != transform_name):
        raise ValueError('手眼结果的格式、mode 或变换名称不一致')
    from .vision.common import checked
    transform = calibration[transform_name]
    checked(transform['matrix'] if isinstance(transform, dict) else transform)
    signature = calibration['signature']
    robot = signature['robot']
    if (signature.get('synthetic') or robot.get('joint_source') != 'measured'
            or robot.get('arm_id') != cfg['arm_id'] or robot.get('parent_frame') != cfg['base_frame']
            or robot.get('firmware') != cfg['firmware']):
        raise ValueError('手眼标定必须对应当前从臂的 measured 来源和基座')
    camera = signature['camera']
    for key in ('serial', 'width', 'height', 'fps'):
        settings.setdefault(key, camera[key])
        if settings[key] != camera[key]:
            raise ValueError(f'视觉配置的 {key} 与手眼标定不一致')
    obj, grasp = object_config(settings['object']), read_json(settings['grasp'])
    if (grasp.get('schema_version') != 1 or grasp.get('kind') != 'object_grasp'
            or grasp.get('synthetic') is not False or grasp.get('usable_for_robot') is not True
            or grasp.get('execution_blocked_reasons')
            or grasp.get('signature') != physical_signature(cfg)
            or grasp.get('object_id') != obj['object_id']
            or grasp.get('mesh_sha256') != digest(obj['mesh_path'])
            or grasp.get('object_frame') != 'original_mesh_scaled_to_m'
            or grasp.get('hand_eye_sha256') != digest(settings['hand_eye'])):
        raise ValueError('抓取绑定未通过检查，或与当前物体、手眼、从臂/TCP 不匹配；重新 bind-grasp')
    checked(grasp['T_object_tcp'])
    return settings


def check_tcp(pose, cfg):
    pose_matrix(pose)
    xyz = np.asarray(pose)[:3]
    if np.any(xyz < cfg['workspace_min_m']) or np.any(xyz > cfg['workspace_max_m']):
        raise ValueError('参考 TCP 超出配置的工作空间')
    for low, high in cfg['keepout_aabbs_m']:
        if np.all(xyz >= low) and np.all(xyz <= high):
            raise ValueError('参考 TCP 位于配置的禁入区域')


def preview_target(capture_dir, observation_path, target, filename):
    """Save and display CAD overlay plus the proposed grasp TCP, then accept/cancel."""
    import cv2
    from .vision.common import read_capture
    from .vision.coordinates import base_color_transforms
    _, metadata, arrays = read_capture(capture_dir)
    obs = read_json(observation_path)
    base_color, _ = base_color_transforms(obs['hand_eye']['path'], metadata)
    color_tcp = np.linalg.inv(base_color[0]) @ pose_matrix(target['pick'])
    view = cv2.imread(str(Path(observation_path).parent/'overlay.png'))
    if view is None:
        raise ValueError('定位叠加图缺失')
    points = np.vstack([np.zeros(3), np.eye(3)*.035])
    points = points @ color_tcp[:3, :3].T + color_tcp[:3, 3]
    if np.all(points[:, 2] > .01):
        uv = points @ arrays['K'].T
        uv = np.rint(uv[:, :2]/uv[:, 2:]).astype(int)
        for end, color in zip(uv[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.arrowedLine(view, tuple(uv[0]), tuple(end), color, 3, tipLength=.2)
        cv2.circle(view, tuple(uv[0]), 7, (0, 255, 255), 2)
        cv2.putText(view, 'PICK TCP', tuple(uv[0]+[10, -10]), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
    xyz = np.asarray(target['pick'])[:3]*1000
    cv2.rectangle(view, (0, 35), (view.shape[1], 115), (30, 30, 30), -1)
    for y, label in ((60, f'Base pick XYZ (mm): {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}'),
                     (95, 'Enter: accept target | C / Esc: cancel | then return to terminal')):
        cv2.putText(view, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    cv2.imwrite(str(filename), view)
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise ValueError(f'视觉采集需要桌面窗口确认目标；预览已保存：{filename}')
    name = 'Visual collection - grasp target'
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, view)
        cv2.resizeWindow(name, min(view.shape[1], 1100), min(view.shape[0], 750))
        while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(30) & 255
            if key in (10, 13):
                return True
            if key in (27, ord('c'), ord('q')):
                return False
        return False
    finally:
        cv2.destroyAllWindows()


class VisualCollection:
    def __init__(self, settings, cfg, output, place_frames=None):
        self.settings, self.cfg, self.output = settings, cfg, Path(output)
        self.target = self.target_path = self.place = self.place_source = None
        self.attempts = self.place_count = 0
        self.directory = new_directory(ROOT/'runs'/f'collect_{self.output.name}_{time.time_ns()}')
        self.robot_config = self.output/'vision_robot_config.json'
        write_json(self.robot_config, cfg)
        write_json(self.output/'vision_config.json', settings)
        paths = [settings[key] for key in ('object', 'hand_eye', 'grasp', 'intrinsics') if settings.get(key)]
        paths += [object_config(settings['object'])['mesh_path'], str(self.robot_config), str(self.output/'vision_config.json')]
        self.sources = {path: digest(path) for path in paths}
        if place_frames:
            document, frames = read_frames(place_frames, cfg)
            if document.get('synthetic') is not False:
                raise ValueError('放置模板必须来自真实记录')
            self.set_place(frames[2], dict(type='frame_file', path=str(Path(place_frames).resolve()), sha256=digest(place_frames)))

    def invalidate(self):
        self.target = self.target_path = None

    def check_sources(self):
        for path, expected in self.sources.items():
            if digest(path) != expected:
                raise ValueError(f'视觉配置/来源文件已改变，请退出后重新启动采集：{path}')

    def set_place(self, pose, source):
        check_tcp(pose, self.cfg)
        self.place = np.asarray(pose, float).copy()
        self.place_count += 1
        path = self.output/f'place_reference_{self.place_count:03d}.json'
        write_json(path, dict(schema_version=1, kind='collection_place_reference', synthetic=False,
                             signature=physical_signature(self.cfg), pose=self.place.tolist(), source=source, created_utc=utc_now()))
        self.place_source = dict(path=str(path), sha256=digest(path), source=source)

    def localize(self, reader):
        # A failed or canceled new observation must never leave the old target usable.
        self.invalidate()
        self.check_sources()
        self.attempts += 1
        attempt = new_directory(self.directory/f'localization_{self.attempts:03d}')
        from .vision.capture import capture
        from .vision.estimate import estimate
        options = dict(self.settings, config=str(self.robot_config), can=self.cfg['channel'], no_robot=False)
        capture_dir = capture(Namespace(**dict(options, out=str(attempt/'capture'))), reader=reader)
        from .vision.coordinates import base_color_transforms
        base_color_transforms(self.settings['hand_eye'], read_json(capture_dir/'capture.json'))
        estimate(Namespace(**dict(options, capture=str(capture_dir), out=str(attempt/'pose'),
                                  mask_file=None, diagnostic_poses=None)))
        obs_path = attempt/'pose/observation.json'
        target = grasp_target(obs_path, self.settings['grasp'], self.cfg)
        reasons = frame_reasons(target)
        if target['synthetic'] or reasons:
            raise ValueError(f'本轮视觉目标未通过检查：{reasons or ["synthetic_target"]}')
        check_tcp(target['pick'], self.cfg)
        print(f'本轮抓取 TCP（m,wxyz）：{target["pick"]}\n检查预览，在窗口按 Enter 接受，C/Esc 取消。', flush=True)
        if not preview_target(capture_dir, obs_path, target, attempt/'pick_preview.png'):
            print('已取消本轮目标；需要时按 V 重新定位。')
            return False
        self.check_sources()
        if frame_reasons(target):
            raise ValueError('目标在预览期间过期或检查失败；按 V 重新定位')
        self.target_path = attempt/'target.json'
        write_json(self.target_path, target)
        self.target_hash = digest(self.target_path)
        self.target = target
        print('本轮视觉目标已接受。保持物体不动，回统一起点后在终端按 R；放置点未记录时先按 L。')
        return True

    def start(self, reader):
        if self.target is None:
            raise ValueError('先按 V 定位并在预览窗口接受本轮抓取目标')
        if self.place is None:
            raise ValueError('先移到放置 TCP 停稳后按 L；后续各轮将复用此放置点')
        self.check_sources()
        if digest(self.target_path) != self.target_hash:
            self.invalidate()
            raise ValueError('视觉目标文件已改变；按 V 重新定位')
        reasons = frame_reasons(self.target)
        if reasons:
            self.invalidate()
            raise ValueError(f'本轮目标已失效，按 V 重新定位：{reasons}')
        source = self.place_source
        if digest(source['path']) != source['sha256']:
            raise ValueError('放置参考文件已改变；重新按 L 记录')
        if source['source'].get('path') and digest(source['source']['path']) != source['source']['sha256']:
            raise ValueError('放置模板文件已改变；重新按 L 记录')
        # Sample after hashing/checking inputs so the first recorded state is fresh.
        state = reader.sample()
        frames = np.array([state['tcp'], self.target['pick'], self.place], float)
        for pose in frames:
            check_tcp(pose, self.cfg)
        document = frame_document(frames, self.cfg, vision=self.target['vision'], execution_blocked_reasons=[],
                                  place_reference=source, vision_collection=dict(source_hashes=self.sources,
                                  target_path=str(self.target_path), target_sha256=self.target_hash))
        path = self.target_path.parent/'frames.json'
        write_json(path, document)
        self.invalidate()  # Every R consumes one observation, including discarded attempts.
        return frames, document, path, state
