from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'


def local_path(value):
    p=Path(value).expanduser()
    p=(p if p.is_absolute() else PROJECT/p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('New outputs and checkpoints must be inside ppo_assembly/')
    return p
