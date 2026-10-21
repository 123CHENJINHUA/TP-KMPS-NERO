"""Object models and grasp profiles stored under assets/objects/<name>."""
import json
from pathlib import Path
import numpy as np

OBJECTS = Path(__file__).resolve().parents[1] / "assets/objects"


def load_object_profile(name):
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Object name must be a simple directory name")
    if name == "cube":
        return None
    directory = OBJECTS / name
    path = directory / "object.json"
    if not path.is_file():
        raise ValueError(f"No object profile found: {path}")
    profile = json.loads(path.read_text())
    if profile.get("name") != name:
        raise ValueError("Object profile name differs from directory")
    asset = (directory / profile["asset"]).resolve()
    if not asset.is_relative_to(directory.resolve()) or not asset.is_file():
        raise ValueError("Object asset must exist inside its model directory")
    dimensions = np.asarray(profile["extent_m"], dtype=float)
    table = np.asarray(profile["table_size_xy_m"], dtype=float)
    numeric = [profile["mass_kg"], profile["grip_width_m"], profile["origin_height_m"]]
    if (dimensions.shape != (3,) or table.shape != (2,)
            or not np.isfinite(numeric).all() or min(numeric) <= 0
            or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)
            or not np.isfinite(table).all() or np.any(table <= dimensions[:2])
            or profile["origin_height_m"] > dimensions[2]):
        raise ValueError("Invalid object dimensions, support height, mass or table")
    return dict(profile, asset_path=str(asset))
