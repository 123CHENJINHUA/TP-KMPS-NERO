from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for directory in (PROJECT, PROJECT / 'pyAgxArm', PROJECT / 'hardware_checks' / '.vendor',
                  PROJECT / 'hand_eye_calibration' / '.vendor', HERE / '.vendor'):
    sys.path.insert(0, str(directory))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))


def inside(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f'输出必须位于独立目录 {HERE}: {path}')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_directory(path):
    path = inside(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
