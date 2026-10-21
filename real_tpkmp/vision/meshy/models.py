"""Make a separate, meter-scale object config; never replace the original CAD."""
import os
from pathlib import Path
import re
import shutil
import subprocess
from ..common import ROOT, new_directory, object_config, output_path, read_json, utc_now, write_json
from ..geometry import load_mesh
from .api import sha256
import numpy as np


def import_model(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.object_id):
        raise ValueError('object-id 只使用英文字母、数字、下划线或短横线')
    source = Path(args.mesh).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in ('.stl', '.glb', '.obj', '.ply'):
        raise ValueError('需要存在的 STL/GLB/OBJ/PLY 三角网格；推荐 GLB 或 STL')
    measured = np.asarray(args.measured_extents_mm, float)
    if not np.isfinite(measured).all() or np.any(measured <= 0):
        raise ValueError('需要以毫米输入三个正的实测尺寸')
    if not np.isfinite(args.rotate_xyz_deg).all():
        raise ValueError('旋转角必须有限')
    python = args.pose_python or os.environ.get('TPKMP_POSE_PYTHON', str(ROOT/'envs/foundationpose/bin/python'))
    if not Path(python).is_file():
        raise ValueError('需要现有 FoundationPose worker 环境的 trimesh；先运行 vision/setup.sh pose-env')
    # A new output directory prevents replacing existing CAD or invalidating old grasps.
    output = new_directory(output_path(args.out or ROOT/'models'/args.object_id))
    snapshot = output/('source'+source.suffix.lower())
    shutil.copyfile(source, snapshot)
    request = dict(source=str(source), source_sha256=sha256(snapshot), output=str(output),
                   measured_extents_mm=measured.tolist(), rotate_xyz_deg=args.rotate_xyz_deg,
                   object_id=args.object_id)
    # OBJ's adjacent MTL/textures stay resolved relative to the original file.
    # GLB/STL/PLY use the preserved snapshot so an external edit cannot race import.
    if source.suffix.lower() != '.obj':
        request['source'] = str(snapshot)
    write_json(output/'import_request.json', request)
    with (output/'import.log').open('x') as log:
        try:
            subprocess.run([python, '-B', str(Path(__file__).with_name('mesh_worker.py')),
                            '--request', str(output/'import_request.json')], check=True, timeout=300,
                           stdout=log, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'模型转换失败，详见 {output / "import.log"}') from exc
    if sha256(source) != request['source_sha256']:
        raise ValueError('导入期间源模型被修改；不生成物体配置，请使用新目录重试')
    report = read_json(output/'geometry.json')
    report.update(schema_version=1, kind='meshy_model_import', created_utc=utc_now(), object_id=args.object_id,
                  source_path=str(source), source_sha256=request['source_sha256'],
                  mesh_sha256=sha256(output/'model.stl'), textured_mesh_sha256=sha256(output/'model.glb'))
    write_json(output/'import.json', report)
    print('均匀缩放后的 XYZ 尺寸 (mm)：'+str((np.asarray(report['normalized_extents_m_xyz'])*1000).round(3).tolist()))
    if not report['dimensions_passed']:
        raise ValueError(f'其余两轴尺寸与实测不符；未生成可用 object.json。查看 {output / "preview.png"}；'
                         '先校正模型朝向或几何，不会强行拉伸模型匹配三轴。')
    cfg = read_json(ROOT/'objects/keel.json')
    for key in ('hsv_lower', 'hsv_upper'):
        cfg.pop(key, None)
    cfg.update(object_id=args.object_id, mesh_path=str((output/'model.stl').relative_to(ROOT)), mesh_units='m',
               expected_extents_m=report['normalized_extents_m_xyz'], mesh_sha256=report['mesh_sha256'],
               text_prompt=args.prompt, model_source='meshy_or_user_import',
               model_import_report=str(output/'import.json'),
               measured_extents_m_unordered=report['measured_extents_m_unordered'],
               notes='Uniform scale from measured longest dimension; other dimensions checked within 10% + 1 mm. '
                     'Generated geometry is approximate. Object frame is the NEW STL frame in meters, centered '
                     'at the rotated AABB center. Teach a new grasp; never reuse the original CAD grasp. '
                     'Pose backend uses STL geometry; normalized GLB retains appearance for viewing only.')
    write_json(output/'object.json', cfg)
    load_mesh(object_config(output/'object.json'))
    print(f'新模型配置：{output / "object.json"}\n尺寸/坐标轴预览：{output / "preview.png"}')
    print('下一步用 estimate.sh --object 此配置做真实 RGB-D 配准，并重新绑定抓取关系。')
    return output
