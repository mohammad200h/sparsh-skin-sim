"""Ensure repo-root imports and paths work when running demos as scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENE_XML = (
    REPO_ROOT
    / "leapXELA_model"
    / "scene_mjx_cube_CoACD_mjx_flex_sensor.xml"
)
