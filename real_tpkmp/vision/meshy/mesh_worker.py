"""CPU mesh conversion in the existing FoundationPose environment, without GPU/CAN."""
import argparse
import json
from pathlib import Path


def normalize(scene, measured_mm, rotation_deg):
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    measured = np.asarray(measured_mm, float)/1000.
    angles = np.asarray(rotation_deg, float)
    if (measured.shape != (3,) or not np.isfinite(measured).all() or np.any(measured <= 0)
            or angles.shape != (3,) or not np.isfinite(angles).all()):
        raise ValueError('需要三个正实测尺寸和有限 XYZ 旋转角')
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or len(mesh.faces) > 2_000_000:
        raise ValueError('需要非空三角网格，最多 200 万面；不是点云')
    if not np.isfinite(mesh.vertices).all() or np.any(np.ptp(mesh.vertices, axis=0) <= 1e-12):
        raise ValueError('模型含无效坐标或零厚度')
    orientation = np.eye(4)
    orientation[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    scene = scene.copy()
    scene.apply_transform(orientation)
    rotated = scene.to_geometry()
    raw_extents = rotated.extents
    # ONE measured length supplies one uniform scale. Never distort individual axes.
    scale = float(measured.max()/raw_extents.max())
    similarity = orientation.copy()
    similarity[:3, :3] *= scale
    similarity[:3, 3] = -rotated.bounds.mean(axis=0)*scale
    centered = np.eye(4)
    centered[:3, :3] *= scale
    centered[:3, 3] = similarity[:3, 3]
    scene.apply_transform(centered)
    mesh = scene.to_geometry()
    if np.any(mesh.area_faces <= 1e-20):
        raise ValueError('模型包含退化三角面，请先在网格编辑器中清理后导入')
    extents = mesh.extents
    difference = abs(np.sort(extents)-np.sort(measured))
    passed = bool(np.all(difference <= .10*np.sort(measured)+.001))
    report = dict(scale_source='measured_longest_extent', uniform_scale_to_m=scale,
                  measured_extents_m_unordered=measured.tolist(), normalized_extents_m_xyz=extents.tolist(),
                  sorted_dimension_errors_m=difference.tolist(), dimensions_passed=passed,
                  dimension_tolerance='10 percent of each measured extent + 1 mm; not a pose precision guarantee',
                  rotation_xyz_deg=angles.tolist(), source_to_object_similarity=similarity.tolist(),
                  object_frame='rotated_mesh_AABB_center_m', vertices=int(len(mesh.vertices)),
                  faces=int(len(mesh.faces)), watertight=bool(mesh.is_watertight),
                  winding_consistent=bool(mesh.is_winding_consistent),
                  note='Bounding dimensions do not verify holes, wall thickness or unseen surfaces. No axis-wise stretching or hole repair performed.')
    return scene, mesh, report


def preview(mesh, destination):
    import numpy as np
    import cv2
    triangles = mesh.triangles
    # Preview only: downsample drawing; exported geometry is unchanged.
    drawing = triangles[np.linspace(0, len(triangles)-1, min(12000, len(triangles)), dtype=int)]
    image = np.full((960, 1120, 3), 248, np.uint8)
    scale = 310./float(mesh.extents.max())
    views = (([.8, -.6, 0], [.3, .4, .8660254], 'Oblique'),
             ([1, 0, 0], [0, 1, 0], 'Top'), ([1, 0, 0], [0, 0, 1], 'Front'),
             ([0, 1, 0], [0, 0, 1], 'Side'))
    for number, (right, up, title) in enumerate(views):
        basis = np.array([right, up], float).T
        normal = np.cross(right, up)
        center = np.array([280+560*(number % 2), 260+430*(number//2)])
        projected = drawing@basis*np.array([scale, -scale])+center
        order = np.argsort(drawing.mean(axis=1)@normal)
        normals = np.cross(drawing[:, 1]-drawing[:, 0], drawing[:, 2]-drawing[:, 0])
        brightness = .5+.5*abs(normals@normal)/np.maximum(np.linalg.norm(normals, axis=1), 1e-20)
        for index in order:
            color = tuple((np.array([190, 150, 90])*brightness[index]).astype(int).tolist())
            cv2.fillConvexPoly(image, np.rint(projected[index]).astype(np.int32), color, lineType=cv2.LINE_AA)
        for label, vector, color in zip('XYZ', np.eye(3)*mesh.extents.max()*.33,
                                         ((30, 30, 230), (30, 160, 30), (230, 70, 30))):
            endpoint = np.rint(vector@basis*np.array([scale, -scale])+center).astype(int)
            if np.linalg.norm(endpoint-center) > 2:
                cv2.arrowedLine(image, tuple(center), tuple(endpoint), color, 2, cv2.LINE_AA, tipLength=.12)
                cv2.putText(image, label, tuple(endpoint+5), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        cv2.putText(image, title, (40+560*(number % 2), 90+430*(number//2)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 30, 30), 2, cv2.LINE_AA)
    lines = ['Uniform measured scale; origin = AABB center; geometry unchanged by preview.',
             'XYZ extents (mm): '+str((mesh.extents*1000).round(2).tolist()),
             'Preview does not validate holes, wall thickness, or pose accuracy.']
    for y, line in zip((30, 900, 935), lines):
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 30), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError('无法保存网格预览')


def main():
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    scene = trimesh.load_scene(request['source'])
    scene, mesh, report = normalize(scene, request['measured_extents_mm'], request['rotate_xyz_deg'])
    directory = Path(request['output'])
    mesh.export(directory/'model.stl', file_type='stl')
    scene.export(directory/'model.glb', file_type='glb')
    preview(mesh, directory/'preview.png')
    (directory/'geometry.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
