"""Persistent configuration for Nero tool cameras and optional environment cameras."""

import json
import re
from pathlib import Path

import numpy as np

from sim_bridge.camera_visuals import visualization_options

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "assets/cameras/wrist_cameras.json"


LENS_KEYS = ("resolution", "focal_length_mm", "horizontal_aperture_mm", "clipping_range_m")


def _validate_lens(config):
    width, height = config["resolution"]
    if any(type(v) is not int or not 16 <= v <= 4096 for v in (width, height)):
        raise ValueError("resolution must contain two integers in [16, 4096]")
    lens = np.asarray([config["focal_length_mm"], config["horizontal_aperture_mm"]], dtype=float)
    clipping = np.asarray(config["clipping_range_m"], dtype=float)
    if lens.shape != (2,) or not np.isfinite(lens).all() or np.any(lens <= 0):
        raise ValueError("lens dimensions must be finite and positive")
    if clipping.shape != (2,) or not np.isfinite(clipping).all() or not 0 < clipping[0] < clipping[1]:
        raise ValueError("clipping range must satisfy 0 < near < far")


def _validate_pose(config, label):
    for name in ("translation_m", "rotation_xyz_deg"):
        vector = np.asarray(config[name], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{label}.{name} must contain three finite values")


def validate_camera_config(config):
    try:
        _validate_lens(config)
        visualization_options(config.get('visualization'))
        for side in ("left", "right"):
            _validate_pose(config[side], side)
            visualization_options(config.get('visualization'), config[side].get('visualization'))
        extras = config.get("extra_cameras", [])
        if not isinstance(extras, list):
            raise ValueError("extra_cameras must be a list")
        paths = {("left", "WristCamera"), ("right", "WristCamera")}
        for item in extras:
            if not isinstance(item, dict):
                raise ValueError("each extra camera must be an object")
            name, parent = item["name"], item["parent"]
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError("camera name must be a USD identifier such as CameraSide")
            if parent not in ("left", "right", "environment"):
                raise ValueError("extra camera parent must be left, right or environment")
            if (parent, name) in paths:
                raise ValueError(f"duplicate camera path: {parent}/{name}")
            paths.add((parent, name))
            _validate_pose(item, name)
            visualization_options(config.get('visualization'), item.get('visualization'))
            _validate_lens({key: item.get(key, config[key]) for key in LENS_KEYS})
            if "show_view" in item and type(item["show_view"]) is not bool:
                raise ValueError("show_view must be true or false")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid wrist camera configuration: {exc}") from exc
    return config


def camera_definitions(config, robot_roots):
    """Keep the two original wrist cameras, then expand optional extra entries."""
    validate_camera_config(config)
    if len(robot_roots) != 2:
        raise ValueError("Wrist cameras require exactly two robot roots")
    parents = dict(left=robot_roots[0] + "/gripper_base",
                   right=robot_roots[1] + "/gripper_base",
                   environment="/World/UserScene/Environment")
    common = {key: config[key] for key in LENS_KEYS}
    definitions = []
    for index, side in enumerate(("left", "right"), 1):
        definitions.append({**common, **config[side], "name": "WristCamera", "parent": parents[side],
                            "window_name": f"Nero {index} Wrist Camera", "arm": index})
    for item in config.get("extra_cameras", []):
        parent = item["parent"]
        arm = {"left": 1, "right": 2, "environment": None}[parent]
        prefix = f"Nero {arm}" if arm else "Environment"
        definitions.append({**common, **item, "window_name": f"{prefix} {item['name']}", "arm": arm})
        definitions[-1]["parent"] = parents[parent]
    for definition in definitions:
        definition["path"] = definition["parent"] + "/" + definition["name"]
        definition["visualization"] = visualization_options(
            config.get("visualization"), definition.get("visualization"))
    return definitions


def load_camera_config(value=DEFAULT_CONFIG):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower() != ".json":
        raise ValueError("Wrist camera configuration must be a JSON inside this project")
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read wrist camera configuration {path}: {exc}") from exc
    return validate_camera_config(config)


def create_wrist_cameras(stage, robot_roots, config, show_views=True):
    """Author local camera poses under the configured tool or environment frames."""
    from pxr import Gf, UsdGeom
    from isaacsim.core.utils.viewports import create_viewport_for_camera

    definitions = camera_definitions(config, robot_roots)
    units = UsdGeom.GetStageMetersPerUnit(stage)
    cameras = []
    for index, definition in enumerate(definitions):
        parent = definition["parent"]
        if not stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError(f"Wrist camera mount frame is missing: {parent}")
        path = definition["path"]
        existing = stage.GetPrimAtPath(path)
        if existing.IsValid() and not existing.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Wrist camera path contains a different prim: {path}")
        camera = UsdGeom.Camera.Define(stage, path)
        transform = UsdGeom.Xformable(camera.GetPrim())
        translation = transform.GetTranslateOp() or transform.AddTranslateOp()
        rotation = transform.GetRotateXYZOp() or transform.AddRotateXYZOp()
        scale = transform.GetScaleOp() or transform.AddScaleOp()
        translation.Set(Gf.Vec3d(*(np.asarray(definition["translation_m"], dtype=float) / units)))
        rotation.Set(Gf.Vec3f(*np.asarray(definition["rotation_xyz_deg"], dtype=float)))
        scale.Set(Gf.Vec3f(1., 1., 1.))
        transform.SetXformOpOrder([translation, rotation, scale], resetXformStack=False)
        camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
        # USD lens dimensions are tenths of a scene unit, not millimeters.
        lens_unit = .001 / (.1 * units)
        aperture = float(definition["horizontal_aperture_mm"]) * lens_unit
        width, height = definition["resolution"]
        camera.CreateFocalLengthAttr().Set(float(definition["focal_length_mm"]) * lens_unit)
        camera.CreateHorizontalApertureAttr().Set(aperture)
        camera.CreateVerticalApertureAttr().Set(aperture * height / width)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*(np.asarray(definition["clipping_range_m"], dtype=float) / units)))
        camera.CreateFStopAttr().Set(0.)
        from sim_bridge.camera_visuals import create_camera_visualization
        visual_path = create_camera_visualization(stage, path, definition["visualization"])
        window = None
        visible = show_views and definition.get("show_view", True)
        if visible:
            window = create_viewport_for_camera(
                definition["window_name"], path, width, height,
                (index % 2) * (width + 15), 60 + (index // 2) * (height + 40))
            window.viewport_api.resolution = (width, height)
        cameras.append(dict(path=path, parent=parent, window=window, definition=definition, visualization=visual_path))
        print(f"WRIST_CAMERA_READY arm={definition['arm']} path={path} "
              f"translation_m={definition['translation_m']} "
              f"rotation_xyz_deg={definition['rotation_xyz_deg']} "
              f"view={visible}", flush=True)
    return cameras
