"""Dependencies and outputs are confined to this independent directory."""
from pathlib import Path
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE / '.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.tmp' / 'matplotlib'))


def inside_folder(path):
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(HERE):
        raise ValueError(f'输出必须位于 {HERE}: {result}')
    return result


def write_json(path, data):
    path = inside_folder(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()
