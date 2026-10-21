"""Export only the user-authored environment from a combined Isaac Sim stage."""

from datetime import datetime
from pathlib import Path

from sim_bridge.app import ROOT

SCENE_ROOT = '/World/UserScene'
SAVE_DIR = ROOT / 'assets' / 'scenes'


def next_scene_snapshot_path():
    return SAVE_DIR / ('workcell_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.usd')


def save_user_scene(stage, destination):
    """Flatten /World/UserScene, including viewport edits, into an environment USD.

    The robot, stand, ground and UI goal prims live outside SCENE_ROOT and are
    therefore excluded. The source stage and its referenced USD are unchanged.
    """
    from pxr import Sdf, Usd, UsdGeom

    path = Path(destination).resolve()
    if not path.is_relative_to(SAVE_DIR) or path.suffix.lower() not in {'.usd', '.usda', '.usdc'}:
        raise ValueError('Environment output must be a USD inside assets/scenes')
    if path.exists():
        raise FileExistsError(f'Environment output already exists: {path}')
    source = stage.GetPrimAtPath(SCENE_ROOT)
    if not source.IsValid():
        raise RuntimeError(f'No editable environment at {SCENE_ROOT}')

    # Compose root and session edits, then reference only the environment prim.
    # Flattening this small stage captures overrides on a referenced environment
    # without serializing the generated robots or stand.
    combined = Sdf.Layer.CreateAnonymous()
    session = stage.GetSessionLayer()
    if session:
        combined.subLayerPaths.append(session.identifier)
    combined.subLayerPaths.append(stage.GetRootLayer().identifier)
    solo = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(solo, UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.SetStageUpAxis(solo, UsdGeom.GetStageUpAxis(stage))
    root = solo.DefinePrim('/World', 'Xform')
    if not root.GetReferences().AddReference(combined.identifier, source.GetPath()):
        raise RuntimeError('Could not compose the environment for export')
    solo.SetDefaultPrim(root)
    flattened = solo.Flatten()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not flattened.Export(str(path)):
        raise RuntimeError(f'Could not save environment USD: {path}')
    saved = Usd.Stage.Open(str(path))
    if saved is None or not saved.GetDefaultPrim().IsValid():
        raise RuntimeError(f'Saved environment USD failed validation: {path}')
    return path
