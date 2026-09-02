"""Re-exec into the repo venv when this interpreter cannot import pyroki.

System Python is pinned to JAX 0.4.14 for DreamerV3. PyRoki needs a newer JAX,
which lives in ``sparsh-skin-sim/.venv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_VENV = _REPO / ".venv"
# Keep the venv shim path. Resolving the symlink points at base CPython and
# drops the venv site-packages (pyvenv.cfg is keyed off argv[0]).
_VENV_PYTHON = _VENV / "bin" / "python"


def _in_project_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == _VENV.resolve()
    except OSError:
        return False


def _pyroki_importable() -> bool:
    try:
        import pyroki  # noqa: F401
        import yourdfpy  # noqa: F401
    except ImportError:
        return False
    return True


def exec_project_venv() -> None:
    """Replace this process with the project venv if pyroki cannot import here."""
    if _in_project_venv():
        return
    if _pyroki_importable():
        return
    if not _VENV_PYTHON.exists():
        return
    python = os.fsdecode(_VENV_PYTHON)
    print(f"Using {python} (system Python cannot import pyroki)", file=sys.stderr)
    os.execv(python, [python, *sys.argv])
