"""Shared camera/vision configuration; no hardware or neural model imports."""
from ..bootstrap import PROJECT, read_json, inside, digest
from .common import ROOT, camera_model, checked, object_config
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import tempfile

DEFAULT_CONFIG = ROOT/'camera_config.json'
PATH_OPTIONS = ('object', 'hand_eye', 'grasp', 'intrinsics', 'sam3_python', 'sam3_checkpoint',
                'sam3_source', 'pose_python', 'foundationpose_root')
CAMERA_OPTIONS = ('serial', 'width', 'height', 'fps', 'intrinsics')
INLINE_OPTIONS = ('intrinsics', 'hand_eye', 'grasp')
SNAPSHOT_ROOT = ROOT/'.cache/config_snapshots'


def parameter_snapshot(value, *, binding_key=None):
    """Persist immutable inline parameters for existing provenance-checked readers."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    expected = hashlib.sha256(payload).hexdigest()
    directory = inside(SNAPSHOT_ROOT/'hand_eye_bindings' if binding_key else SNAPSHOT_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{binding_key or expected}.json'
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            try:
                os.link(temporary, path)  # Publish only a complete document; never overwrite.
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    if digest(path) != expected:
        if binding_key:
            raise ValueError('该手眼矩阵绑定的相机/内参或机械臂已变化；请重新标定验证后填写新矩阵')
        raise ValueError('配置参数快照被修改；不能继续使用该来源')
    return str(path)


def camera_parameters(settings):
    value = settings.get('intrinsics')
    if isinstance(value, str):
        value = read_json(value)
    if not isinstance(value, dict):
        raise ValueError('简化手眼配置需要在 intrinsics 中直接填写对应的 K 和畸变参数')
    model = dict(value)
    for key in ('serial', 'width', 'height', 'fps'):
        if key in settings:
            model.setdefault(key, settings[key])
    model.setdefault('frame', 'color_optical')
    camera_model(model)
    return {key: model[key] for key in ('serial', 'width', 'height', 'fps', 'frame', 'K',
                                       'distortion_model', 'distortion_coefficients')}


def expand_hand_eye(value, settings, cfg=None):
    """Generate runtime metadata from the shared settings instead of duplicating user fields."""
    if cfg is None:
        from ..data import config
        cfg = config()
    mode = value.get('mode')
    name = {'eye-in-hand': 'T_flange_color', 'eye-to-hand': 'T_base_color'}.get(mode)
    if name is None or value.get('translation_unit', 'm') != 'm':
        raise ValueError('手眼 mode 需要 eye-in-hand/eye-to-hand，矩阵平移单位为米')
    matrix = checked(value['matrix']).tolist()
    signature = dict(robot=dict(arm_id=cfg['arm_id'], firmware=cfg['firmware'],
                               parent_frame=cfg['base_frame'], joint_source='measured'),
                     camera=camera_parameters(settings), synthetic=bool(value.get('synthetic', False)))
    passed = value.get('passed', False)
    if type(passed) is not bool:
        raise ValueError('passed 需要填写实际验证结果 true/false')
    if passed and not signature['synthetic']:
        # Keep the calibration association outside the editable file. Reusing a matrix
        # after changing the camera/geometry must still fail even though those fields occur once.
        key = hashlib.sha256(json.dumps(dict(mode=mode, matrix=matrix), sort_keys=True).encode()).hexdigest()
        parameter_snapshot(signature, binding_key=key)
    return dict(schema_version=1, mode=mode, transform_name=name, **{name: dict(matrix=matrix, translation_unit='m')},
                passed=passed, synthetic=signature['synthetic'], signature=signature)


def materialize_settings(settings, names=INLINE_OPTIONS, cfg=None):
    """Adapt inline parameter objects without changing any validation/pass flags."""
    result = dict(settings)
    for name in names:
        value = result.get(name)
        if not isinstance(value, dict):
            continue
        document = copy.deepcopy(value)
        try:
            if name == 'intrinsics':
                document.update(camera_parameters(dict(result, intrinsics=document)))
                for key in ('serial', 'width', 'height', 'fps'):
                    if document.get(key) is None:
                        raise ValueError(f'内参参数缺少 {key}')
                document['source'] = 'inline_parameters'
            elif name == 'hand_eye':
                if 'matrix' in document and 'schema_version' not in document:
                    document = expand_hand_eye(document, result, cfg)
                transform_name = {'eye-in-hand': 'T_flange_color', 'eye-to-hand': 'T_base_color'}.get(document.get('mode'))
                if document.get('schema_version') != 1 or transform_name is None or document.get('transform_name') != transform_name:
                    raise ValueError('手眼 mode 和 transform_name 不一致')
                value = document[transform_name]
                checked(value['matrix'] if isinstance(value, dict) else value)
                if isinstance(value, dict) and value.get('translation_unit', 'm') != 'm':
                    raise ValueError('手眼矩阵平移单位必须为 m')
                signature = document['signature']
                camera_model(signature['camera'])
                for key in ('serial', 'width', 'height', 'fps'):
                    if signature['camera'].get(key) is None:
                        raise ValueError(f'标定相机信息缺少 {key}')
                for key in ('arm_id', 'firmware', 'parent_frame', 'joint_source'):
                    if not signature['robot'].get(key):
                        raise ValueError(f'标定机械臂信息缺少 {key}')
                if type(document.get('passed')) is not bool:
                    raise ValueError('passed 需要填写实际验证结果 true/false')
            elif name == 'grasp':
                if document.get('schema_version') != 1 or document.get('kind') != 'object_grasp':
                    raise ValueError('需要完整的 object_grasp 参数对象')
                checked(document['T_object_tcp'])
                required = {'signature', 'object_id', 'mesh_sha256', 'object_frame', 'hand_eye_sha256',
                            'synthetic', 'usable_for_robot', 'execution_blocked_reasons'}
                if required-document.keys():
                    raise ValueError(f'抓取绑定信息缺少 {sorted(required-document.keys())}；可用 bind-grasp --update-config 自动填写')
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'请补齐相机配置中的 {name} 参数：{exc}') from exc
        result[name] = parameter_snapshot(document)
    return result


def update_grasp_parameters(path, grasp, cfg=None):
    """Store a newly verified binding directly in the user's parameter JSON."""
    if grasp.get('synthetic') or not grasp.get('usable_for_robot') or grasp.get('execution_blocked_reasons'):
        raise ValueError('只有通过检查的真实抓取绑定才能写入相机参数配置')
    path = inside(path)
    original_hash = digest(path)
    document = read_json(path)
    settings = materialize_settings(read_settings(path), names=('hand_eye',), cfg=cfg)
    if not settings['hand_eye'] or digest(settings['hand_eye']) != grasp['hand_eye_sha256']:
        raise ValueError('本次绑定的手眼参数与相机配置不一致；未更新配置')
    obj = object_config(settings['object'])
    if grasp['object_id'] != obj['object_id'] or grasp['mesh_sha256'] != digest(obj['mesh_path']):
        raise ValueError('本次绑定的物体与相机配置不一致；未更新配置')
    fields = ('schema_version', 'kind', 'object_id', 'mesh_sha256', 'object_frame', 'T_object_tcp',
              'signature', 'synthetic', 'hand_eye_sha256', 'execution_blocked_reasons', 'usable_for_robot')
    document['grasp'] = {key: grasp[key] for key in fields}
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    try:
        if digest(path) != original_hash:
            raise ValueError('写入期间相机配置被其他程序修改；未更新配置')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_settings(path=None):
    path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    try:
        supplied = read_json(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取相机配置 {path}：{exc}') from exc
    settings = dict(object=str(ROOT/'objects/keel.json'), hand_eye=None, grasp=None, intrinsics=None,
                    frames=3, mask_method='box', box=None, prompt=None, instance=None,
                    sam3_python=None, sam3_checkpoint=None, sam3_source=None, pose_python=None,
                    foundationpose_root=None, iterations=5)
    allowed = set(settings) | {'serial', 'width', 'height', 'fps', 'object_id'}
    if not isinstance(supplied, dict):
        raise ValueError(f'相机配置必须是 JSON 对象：{path}')
    if set(supplied)-allowed:
        raise ValueError(f'相机配置含未知字段：{sorted(set(supplied)-allowed)}')
    settings.update(supplied)
    if 'object_id' in supplied:
        name = settings.pop('object_id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
            raise ValueError('object_id 使用字母开头的物体名称，可包含数字、下划线和连字符')
        if supplied.get('object'):
            raise ValueError('object_id 与旧版 object 路径不能同时填写')
        candidates = [ROOT/'objects'/f'{name}.json', ROOT/'models'/name/'object.json']
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if not matches:
            raise ValueError(f'未找到物体 {name}；先在 objects/ 或 models/ 中注册该模型')
        if read_json(matches[0]).get('object_id') != name:
            raise ValueError('物体名称与模型配置的 object_id 不一致')
        settings['object'] = str(matches[0])
    for key in PATH_OPTIONS:
        value = settings.get(key)
        if key in INLINE_OPTIONS and isinstance(value, dict):
            continue
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} 需要参数对象；未设置时填写 null（兼容旧版路径字符串）')
            value = Path(value).expanduser()
            settings[key] = str((value if value.is_absolute() else PROJECT/value).resolve())
    if 'serial' in settings and (not isinstance(settings['serial'], str) or not settings['serial'].strip()):
        raise ValueError('serial 需要相机序列号字符串，例如 "213522070575"')
    for key in ('width', 'height', 'fps', 'frames', 'iterations'):
        if key in settings and (type(settings[key]) is not int or settings[key] <= 0):
            raise ValueError(f'{key} 必须为正整数')
    if isinstance(settings['intrinsics'], dict):
        # Legacy full camera objects retain their identity checks; compact objects inherit it once.
        for key in ('serial', 'width', 'height', 'fps'):
            if key in settings:
                settings['intrinsics'].setdefault(key, settings[key])
        settings['intrinsics'].setdefault('frame', 'color_optical')
    if settings['frames'] < 3 or settings['mask_method'] not in ('box', 'sam3', 'hsv'):
        raise ValueError('至少采集 3 帧；mask_method 需要 box、sam3 或 hsv')
    if settings['box'] is not None:
        from .box_prompt import checked_box
        checked_box(settings['box'])
    return settings


def apply_settings(args):
    """Fill omitted CLI options from JSON; explicit flags always take precedence."""
    command = args.command
    if command == 'track' and args.replay:
        # Replay uses recorded camera geometry and remains independent of live configuration.
        args.mask_method = args.mask_method or 'box'
        return
    settings = read_settings(args.camera_config)
    fields = {
        'capture': (*CAMERA_OPTIONS, 'frames'),
        'track': (*CAMERA_OPTIONS, 'object', 'mask_method', 'box', 'prompt', 'instance',
                  'sam3_python', 'sam3_checkpoint', 'sam3_source', 'pose_python', 'foundationpose_root'),
        'estimate': ('hand_eye', 'object', 'mask_method', 'box', 'prompt', 'instance', 'iterations',
                     'sam3_python', 'sam3_checkpoint', 'sam3_source', 'pose_python', 'foundationpose_root'),
        'frames': ('grasp',),
    }[command]
    for name in fields:
        if getattr(args, name, None) is None:
            setattr(args, name, settings.get(name))
    if getattr(args, 'factory_intrinsics', False):
        args.intrinsics = None
    if getattr(args, 'camera_only', False):
        args.hand_eye = None
    inline = {name: getattr(args, name) for name in INLINE_OPTIONS if hasattr(args, name)}
    if isinstance(inline.get('intrinsics'), dict) and command in ('capture', 'track'):
        for name in ('serial', 'width', 'height', 'fps'):
            if inline['intrinsics'].get(name) != getattr(args, name):
                raise ValueError(f'内参参数的 {name} 与本次相机配置不一致')
    context = dict(settings, **inline)
    cfg = None
    if isinstance(inline.get('hand_eye'), dict) and 'matrix' in inline['hand_eye']:
        # The observation belongs to the robot configuration saved at capture time.
        # Do not infer its identity from today's default robot configuration.
        metadata = read_json(Path(args.capture)/'capture.json')
        cfg = metadata.get('config')
        if not isinstance(cfg, dict) or any(key not in cfg for key in ('arm_id', 'firmware', 'base_frame')):
            raise ValueError('RGB-D 采集记录缺少机械臂身份配置')
    resolved = materialize_settings(context, names=inline, cfg=cfg)
    for name in inline:
        setattr(args, name, resolved[name])
    if command in ('capture', 'track'):
        for name in ('serial', 'width', 'height', 'fps'):
            if getattr(args, name) is None:
                raise ValueError(f'请在相机配置中填写 {name}，或用命令行参数指定')
    if command == 'frames' and not args.grasp:
        raise ValueError('请在相机配置的 grasp 字段填写抓取绑定参数，或指定 --grasp')
    print(f'读取相机配置：{Path(args.camera_config).expanduser().resolve()}', flush=True)
