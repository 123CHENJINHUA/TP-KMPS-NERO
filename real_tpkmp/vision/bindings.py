"""Teach object<-TCP once, then generate new visual pick task frames."""
from .common import checked, output_path, write_json, read_json, digest, utc_now
from ..data import config, physical_signature, read_record, read_frames, frame_document
from ..kinematics import pose_matrix, matrix_pose
from pathlib import Path
import time
import numpy as np


def observation(path):
    result = read_json(path)
    if result.get('schema_version') != 1 or result.get('kind') != 'object_observation':
        raise ValueError('需要视觉 observation.json')
    if result.get('T_base_object') is None:
        raise ValueError('只有相机坐标系的物体位姿；需要指定同组内参对应的手眼标定结果')
    checked(result['T_base_object'])
    if not result.get('hand_eye'):
        raise ValueError('缺少手眼标定来源')
    if digest(result['hand_eye']['path']) != result['hand_eye']['sha256']:
        raise ValueError('手眼标定文件发生变化；需重新定位并绑定抓取关系')
    return result


def bind(args):
    obs = observation(args.observation)
    cfg = config(args.config, args.can)
    if physical_signature(cfg) != physical_signature(obs['config']):
        raise ValueError('物体定位与抓取示教的实体臂/TCP 配置不匹配')
    synthetic = obs['synthetic']
    if args.demo:
        meta, data = read_record(args.demo, 'demonstration')
        if meta.get('joint_source') != 'measured' or meta['signature'] != physical_signature(cfg):
            raise ValueError('示教必须来自同一从臂和相同工具配置的 measured 数据')
        pose = data['frames'][1]
        synthetic |= meta['synthetic']
        source = dict(type='demonstration_pick_reference', path=str(Path(args.demo).resolve()), sha256=digest(args.demo))
    else:
        from ..hardware import Reader
        from ..terminal import keys
        reader = Reader(cfg)
        try:
            reader.wait_ready(False)
            print('保持物体与参考定位时的位置不变。拖动主臂把从臂 TCP 放到实际抓取位姿；停稳后按空格记录，Q 退出。')
            print('此步骤只读取反馈；不要在记录前把物体抓走。')
            with keys() as key:
                while True:
                    pressed = key()
                    if pressed in ('q', '\x1b'):
                        return
                    if pressed == ' ':
                        states = []
                        for _ in range(15):
                            states.append(reader.sample(False))
                            time.sleep(.02)
                        if np.max(np.ptp([s['joints_rad'] for s in states], axis=0)) > .005:
                            print('从臂尚未停稳，请重试。')
                            continue
                        pose = states[-1]['tcp']
                        source = dict(type='live_measured', joints_rad=states[-1]['joints_rad'].tolist(), channel=cfg['channel'])
                        break
                    time.sleep(.02)
        finally:
            reader.close()
    T_object_tcp = np.linalg.inv(checked(obs['T_base_object']))@pose_matrix(pose)
    reasons = list(obs['execution_blocked_reasons'])
    if synthetic:
        reasons.append('synthetic_grasp_binding')
    if np.linalg.norm(T_object_tcp[:3, 3]) > 1.:
        raise ValueError('抓取 TCP 离物体超过 1m；检查坐标系、单位和 TCP 定义')
    result = dict(schema_version=1, kind='object_grasp', created_utc=utc_now(),
                  object_id=obs['object_id'], mesh_sha256=obs['mesh_sha256'], object_frame=obs['object_frame'],
                  T_object_tcp=T_object_tcp.tolist(), signature=physical_signature(cfg), synthetic=bool(synthetic),
                  reference_observation=str(Path(args.observation).resolve()), reference_observation_sha256=digest(args.observation),
                  hand_eye_sha256=obs['hand_eye']['sha256'], source=source,
                  execution_blocked_reasons=sorted(set(reasons)), usable_for_robot=not reasons)
    write_json(output_path(args.out), result)
    print(f'保存物体到抓取 TCP 的变换：{args.out}\n{T_object_tcp}\n可用于真机={result["usable_for_robot"]}')
    return result


def grasp_target(obs_path, grasp_path, cfg):
    """Compose a visual pick without requiring an existing demonstration/template."""
    obs, grasp = observation(obs_path), read_json(grasp_path)
    if grasp.get('schema_version') != 1 or grasp.get('kind') != 'object_grasp':
        raise ValueError('需要 bind-grasp 生成的抓取关系 JSON')
    if (grasp['signature'] != physical_signature(cfg) or physical_signature(obs['config']) != physical_signature(cfg)
            or grasp['object_id'] != obs['object_id'] or grasp['mesh_sha256'] != obs['mesh_sha256']
            or grasp['object_frame'] != obs['object_frame'] or grasp['hand_eye_sha256'] != obs['hand_eye']['sha256']):
        raise ValueError('抓取关系与当前物体 CAD、实体臂/TCP 或手眼标定不匹配；需重新绑定')
    pick = matrix_pose(checked(obs['T_base_object'])@checked(grasp['T_object_tcp']))
    reasons = list(obs['execution_blocked_reasons'])+list(grasp['execution_blocked_reasons'])
    if not obs.get('usable_for_robot') or not grasp.get('usable_for_robot'):
        reasons.append('vision_or_grasp_not_validated')
    age = time.time()-obs['observed_unix_s']
    if age < -1 or age > obs['max_observation_age_s']:
        reasons.append('observation_expired')
    synthetic = bool(obs['synthetic'] or grasp['synthetic'])
    return dict(schema_version=1, kind='visual_pick_target', pick=pick.tolist(),
                signature=physical_signature(cfg), synthetic=synthetic,
                execution_blocked_reasons=sorted(set(reasons)),
                vision=dict(observation_path=str(Path(obs_path).resolve()), observation_sha256=digest(obs_path),
                                          grasp_path=str(Path(grasp_path).resolve()), grasp_sha256=digest(grasp_path),
                                          object_id=obs['object_id'], observed_unix_s=obs['observed_unix_s'],
                                          max_observation_age_s=obs['max_observation_age_s']))


def build_frames(obs_path, grasp_path, template_path, cfg):
    target = grasp_target(obs_path, grasp_path, cfg)
    template, frames = read_frames(template_path, cfg)
    frames[1] = target['pick']
    return frame_document(frames, cfg, target['synthetic'] or template.get('synthetic', True),
                          execution_blocked_reasons=target['execution_blocked_reasons'], vision=target['vision'],
                          template_path=str(Path(template_path).resolve()))


def frame_reasons(document):
    reasons = list(document.get('execution_blocked_reasons', []))
    evidence = document.get('vision')
    if evidence:
        for key in ('observation', 'grasp'):
            if digest(evidence[key+'_path']) != evidence[key+'_sha256']:
                raise ValueError(f'视觉 {key} 来源文件被修改；重新生成任务参考系')
        obs = observation(evidence['observation_path'])
        grasp = read_json(evidence['grasp_path'])
        if not obs.get('usable_for_robot') or not grasp.get('usable_for_robot'):
            reasons.append('vision_evidence_not_validated')
        age = time.time()-obs['observed_unix_s']
        if age < -1 or age > obs['max_observation_age_s']:
            reasons.append('observation_expired')
    return sorted(set(reasons))


def export_frames(args):
    cfg = config(args.config)
    document = build_frames(args.observation, args.grasp, args.template_frames, cfg)
    write_json(output_path(args.out), document)
    print(f'已生成视觉任务参考系：{args.out}\n抓取 TCP：{document["frames"]["pick"]}')
    print(f'执行阻止原因：{document["execution_blocked_reasons"]}；起点和放置位姿沿用模板。')
