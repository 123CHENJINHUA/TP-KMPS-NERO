"""Single-key logging controls, always restore terminal after an exception."""
import select
import sys
import termios
import tty
from contextlib import contextmanager


def flush_keys():
    """Discard keys typed during blocking localization/point capture."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


@contextmanager
def keys():
    if not sys.stdin.isatty():
        raise ValueError('请在交互终端运行采集/参考系记录脚本')
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield lambda: sys.stdin.read(1).lower() if select.select([sys.stdin], [], [], 0)[0] else ''
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
