"""Save and restore Isaac window arrangements without depending on layout menus."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT = ROOT / 'assets/layouts/dual_camera_layout.json'


def resolve_layout_path(value=DEFAULT_LAYOUT):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower() != '.json':
        raise ValueError('UI layout must be a JSON file inside this project')
    return path


def save_layout(destination=DEFAULT_LAYOUT):
    import omni.ui as ui

    path = resolve_layout_path(destination)
    workspace = ui.Workspace.dump_workspace()
    if not isinstance(workspace, list) or not workspace:
        raise RuntimeError('No window layout is available to save')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workspace, ensure_ascii=False, indent=2) + '\n')
    print(f'UI_LAYOUT_SAVED path={path}', flush=True)
    return path


def load_layout(source=DEFAULT_LAYOUT):
    import omni.ui as ui

    path = resolve_layout_path(source)
    if not path.is_file():
        raise FileNotFoundError(f'No saved layout at {path}; use Save UI Layout first')
    workspace = json.loads(path.read_text())
    if not isinstance(workspace, list) or not workspace or not all(isinstance(item, dict) for item in workspace):
        raise ValueError('The file does not contain an Isaac window layout')
    # Keep new camera/control windows open when loading an older arrangement.
    ui.Workspace.restore_workspace(workspace, keep_windows_open=True)
    print(f'UI_LAYOUT_LOAD_REQUESTED path={path}', flush=True)
    return path


def build_layout_controls():
    """Build buttons inside the caller's control-panel VStack."""
    import omni.ui as ui

    status = ui.SimpleStringModel('Layout: assets/layouts/dual_camera_layout.json')

    def perform(action):
        try:
            path = action()
        except Exception as exc:
            status.set_value(f'Layout failed: {exc}')
            print(f'UI_LAYOUT_FAILED {exc}', flush=True)
        else:
            prefix = 'Saved layout' if action is save_layout else 'Loading layout'
            status.set_value(f'{prefix}: {path.relative_to(ROOT)}')

    with ui.HStack(height=30, spacing=8):
        ui.Button('Save UI Layout', clicked_fn=lambda: perform(save_layout))
        ui.Button('Load UI Layout', clicked_fn=lambda: perform(load_layout))
    ui.StringField(model=status, read_only=True, height=35, multiline=True)
    return status
