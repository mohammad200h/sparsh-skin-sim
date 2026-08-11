"""Helpers for adding objects to a MuJoCo scene (via ``MjSpec``)."""

from __future__ import annotations

from typing import Callable, Sequence

import mujoco as mj
import numpy as np

from .flex_util import flex_id, flex_vertex_body_ids
from .fk_taxel_util import flex_vertex_for_fk_grid

# Match the former reorientation-cube defaults.
DEFAULT_CUBE_NAME = "cube"
DEFAULT_CUBE_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_CUBE_HALF_SIZE = 0.0385
DEFAULT_CUBE_MASS = 0.108
DEFAULT_CUBE_RGBA = (0.85, 0.45, 0.15, 1.0)
DEFAULT_CUBE_CLEARANCE = 0.02  # gap above palm sensors when spawning
# Small cube dropped on a single taxel; mass is high for the volume.
DEFAULT_TAXEL_CUBE_HALF_SIZE = 0.005  # 2.5 mm cube (was 1 cm / 4)
DEFAULT_TAXEL_CUBE_MASS = 0.01  # kg (was 0.5 × 4)
DEFAULT_TAXEL_CUBE_RGBA = (0.95, 0.25, 0.20, 1.0)
DEFAULT_TAXEL_CUBE_CLEARANCE = 0.03  # gap above taxel surface when spawning
# Sticky cube welded to a taxel vertex (fingertip probe defaults).
DEFAULT_STICKY_CUBE_HALF_SIZE = 0.008 / 3.5  # ~2.3 mm half-extent (~4.6 mm cube)
DEFAULT_STICKY_CUBE_MASS = 0.02  # kg
DEFAULT_STICKY_CUBE_RGBA = (0.20, 0.75, 0.35, 1.0)
DEFAULT_STICKY_CUBE_CLEARANCE = 0.0  # cube just touches the taxel surface
# Regular pad grids: (n_cols, n_rows), matching SensorDefinition.count[:2].
# Fingertip pads use a sparse 6×6 FK canvas (30 active taxels).
FLEX_GRID_SIZES: dict[str, tuple[int, int]] = {
    "uspa46": (6, 4),
    "uspa44": (4, 4),
    "tip": (6, 6),
}
# Bit 1 for rigid contacts; affinity bit 2 so the cube collides with flex skins
# (flex contype/conaffinity = 2) and the floor (also bit 2).
DEFAULT_CUBE_CONTYPE = 1
DEFAULT_CUBE_CONAFFINITY = 2
PALM_FLEX_SUBSTRING = "uspa46"

# Tetromino cell size (half-extent of each unit cube). Smaller than the
# reorientation cube so a 4-block piece still fits over the palm.
DEFAULT_TETRIS_BLOCK_HALF = 0.012
DEFAULT_TETRIS_BLOCK_MASS = 0.02
DEFAULT_TETRIS_CLEARANCE = 0.02
DEFAULT_TETRIS_SPACING = 0.06  # centre-to-centre gap when placing multiple

# Classic 7 tetrominoes as (x, y) cell offsets in the horizontal plane.
# Cell units are later scaled by ``2 * block_half``.
TETRIS_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "L": ((0, 0), (0, 1), (0, 2), (1, 2)),
    "J": ((1, 0), (1, 1), (1, 2), (0, 2)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
}
TETRIS_SHAPE_NAMES: tuple[str, ...] = tuple(TETRIS_SHAPES.keys())
TETRIS_COLORS: dict[str, tuple[float, float, float, float]] = {
    "I": (0.20, 0.80, 0.90, 1.0),  # cyan
    "O": (0.95, 0.85, 0.15, 1.0),  # yellow
    "T": (0.65, 0.30, 0.85, 1.0),  # purple
    "L": (0.95, 0.55, 0.15, 1.0),  # orange
    "J": (0.20, 0.35, 0.90, 1.0),  # blue
    "S": (0.25, 0.80, 0.35, 1.0),  # green
    "Z": (0.90, 0.25, 0.25, 1.0),  # red
}


def palm_sensor_center(model: mj.MjModel, data: mj.MjData | None = None) -> np.ndarray:
    """World-frame centre of all palm (``uspa46``) flex vertices."""
    if data is None:
        data = mj.MjData(model)
        mj.mj_forward(model, data)

    centres: list[np.ndarray] = []
    for flex_id in _matching_flex_ids(model, flex_substring=PALM_FLEX_SUBSTRING):
        adr = int(model.flex_vertadr[flex_id])
        n = int(model.flex_vertnum[flex_id])
        centres.append(np.asarray(data.flexvert_xpos[adr : adr + n]).mean(axis=0))

    if not centres:
        raise ValueError(
            f"No palm flexes matching '{PALM_FLEX_SUBSTRING}' found in the model"
        )
    return np.mean(np.stack(centres, axis=0), axis=0)


def _list_flex_names(model: mj.MjModel) -> list[str]:
    names: list[str] = []
    for flex_id in range(model.nflex):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, flex_id)
        if name is not None:
            names.append(name)
    return names


def _normalize_flex_name(flex_name: str) -> str:
    name = flex_name.strip()
    if not name:
        raise ValueError("flex_name must be a non-empty string")
    return name if name.startswith("flex_") else f"flex_{name}"


def _matching_flex_ids(
    model: mj.MjModel,
    *,
    flex_name: str | None = None,
    flex_substring: str | None = None,
) -> list[int]:
    """Resolve flex ids by exact name (with optional ``flex_`` prefix) or substring."""
    if flex_name is not None and flex_substring is not None:
        raise ValueError("Pass only one of flex_name or flex_substring, not both")
    if flex_name is None and flex_substring is None:
        raise ValueError("flex_name or flex_substring is required")

    available = _list_flex_names(model)
    if flex_name is not None:
        candidates = [flex_name.strip(), _normalize_flex_name(flex_name)]
        for candidate in candidates:
            fid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, candidate)
            if fid >= 0:
                return [int(fid)]
        raise ValueError(
            f"Flex '{flex_name}' not found. Available: {', '.join(available)}"
        )

    assert flex_substring is not None
    matched = [
        int(fid)
        for fid in range(model.nflex)
        if (name := mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, fid)) is not None
        and flex_substring in name
    ]
    if not matched:
        raise ValueError(
            f"No flexes matching substring '{flex_substring}'. "
            f"Available: {', '.join(available)}"
        )
    return matched


def spawn_pose_above_flex(
    spec: mj.MjSpec,
    *,
    flex_name: str | None = None,
    flex_substring: str | None = None,
    half_size: float | Sequence[float] = DEFAULT_CUBE_HALF_SIZE,
    clearance: float = DEFAULT_CUBE_CLEARANCE,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Object centre above one or more flex pads, plus an optional XYZ offset.

    Compiles ``spec`` temporarily to read flex vertex positions, then returns
    ``[mean_x, mean_y, max_z + half_extent_z + clearance] + offset``.

    Provide either ``flex_name`` (exact, ``flex_`` prefix optional) or
    ``flex_substring`` (e.g. ``uspa46`` for all palm pads).
    """
    size = np.asarray(half_size, dtype=np.float64).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 3)
    elif size.size != 3:
        raise ValueError(f"half_size must be a scalar or length-3, got {half_size!r}")

    delta = np.asarray(offset, dtype=np.float64).reshape(-1)
    if delta.size != 3:
        raise ValueError(f"offset must be length-3 (xyz), got {offset!r}")

    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    tops: list[float] = []
    xy: list[np.ndarray] = []
    for flex_id in _matching_flex_ids(
        model, flex_name=flex_name, flex_substring=flex_substring
    ):
        adr = int(model.flex_vertadr[flex_id])
        n = int(model.flex_vertnum[flex_id])
        verts = np.asarray(data.flexvert_xpos[adr : adr + n])
        xy.append(verts.mean(axis=0)[:2])
        tops.append(float(verts[:, 2].max()))

    centre_xy = np.mean(np.stack(xy, axis=0), axis=0)
    z = max(tops) + float(size[2]) + float(clearance)
    return np.array([centre_xy[0], centre_xy[1], z], dtype=np.float64) + delta


def infer_flex_grid_size(flex_name: str) -> tuple[int, int]:
    """Return ``(n_cols, n_rows)`` for a flex pad FK grid.

    Examples: ``uspa46`` → ``(6, 4)``, ``uspa44`` → ``(4, 4)``,
    ``mf_tip`` → ``(6, 6)`` (sparse fingertip canvas).
    """
    normalized = _normalize_flex_name(flex_name)
    # Prefer specific pad patterns over the shared ``tip`` substring.
    for key in ("uspa46", "uspa44", "tip"):
        if key in normalized:
            return FLEX_GRID_SIZES[key]
    raise ValueError(
        f"Cannot infer grid size for flex '{flex_name}'. "
        f"Known patterns: {', '.join(FLEX_GRID_SIZES)}. "
        "Pass ``grid_size=(n_cols, n_rows)`` explicitly."
    )


def taxel_vertex_body_name(
    spec: mj.MjSpec,
    *,
    flex_name: str,
    grid_row: int,
    grid_col: int,
) -> tuple[str, int, int]:
    """Resolve FK grid cell → ``(vertex_body_name, vertex_index, taxel_id)``."""
    normalized = _normalize_flex_name(flex_name)
    model = spec.compile()
    vertex_index, taxel_id = flex_vertex_for_fk_grid(
        model, normalized, grid_row, grid_col
    )
    body_ids = flex_vertex_body_ids(model, normalized)
    body_id = int(body_ids[vertex_index])
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    if body_name is None:
        raise ValueError(
            f"Flex '{normalized}' vertex {vertex_index} has no named body"
        )
    return body_name, int(vertex_index), int(taxel_id)


def flex_grid_vertex_index(
    grid_row: int,
    grid_col: int,
    *,
    n_cols: int,
    n_rows: int,
) -> int:
    """Map a taxel grid cell to the flex vertex index (MuJoCo grid order)."""
    if not (0 <= grid_row < n_rows):
        raise ValueError(
            f"grid_row must be in [0, {n_rows}), got {grid_row}"
        )
    if not (0 <= grid_col < n_cols):
        raise ValueError(
            f"grid_col must be in [0, {n_cols}), got {grid_col}"
        )
    return int(grid_col * n_rows + grid_row)


def flex_taxel_pose(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
    vertex_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame position and outward normal (+z) for one flex vertex."""
    fid = flex_id(model, flex_name)
    adr = int(model.flex_vertadr[fid])
    n = int(model.flex_vertnum[fid])
    if not (0 <= vertex_index < n):
        raise ValueError(
            f"vertex_index must be in [0, {n}), got {vertex_index} "
            f"for flex '{flex_name}'"
        )

    position = np.asarray(
        data.flexvert_xpos[adr + vertex_index], dtype=np.float64
    )
    body_ids = flex_vertex_body_ids(model, flex_name)
    body_id = int(body_ids[vertex_index])
    rotation = data.xmat[body_id].reshape(3, 3)
    normal = rotation[:, 2]
    return position, normal


def flex_taxel_frame(
    model: mj.MjModel,
    data: mj.MjData,
    flex_name: str,
    vertex_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame position and orientation quat (wxyz) for one flex vertex."""
    fid = flex_id(model, flex_name)
    adr = int(model.flex_vertadr[fid])
    n = int(model.flex_vertnum[fid])
    if not (0 <= vertex_index < n):
        raise ValueError(
            f"vertex_index must be in [0, {n}), got {vertex_index} "
            f"for flex '{flex_name}'"
        )

    position = np.asarray(
        data.flexvert_xpos[adr + vertex_index], dtype=np.float64
    )
    body_ids = flex_vertex_body_ids(model, flex_name)
    body_id = int(body_ids[vertex_index])
    quat = np.empty(4, dtype=np.float64)
    mj.mju_mat2Quat(quat, data.xmat[body_id])
    return position, quat


def spawn_pose_above_taxel(
    spec: mj.MjSpec,
    *,
    flex_name: str,
    grid_row: int,
    grid_col: int,
    grid_size: tuple[int, int] | None = None,
    half_size: float | Sequence[float] = DEFAULT_TAXEL_CUBE_HALF_SIZE,
    clearance: float = DEFAULT_TAXEL_CUBE_CLEARANCE,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    configure: Callable[[mj.MjModel, mj.MjData], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cube centre and orientation above one taxel.

    Returns
    -------
    pos, quat:
        World-frame centre and taxel-aligned quaternion (wxyz). ``offset`` is
        applied in the taxel-local frame (same convention as sticky cubes).
    """
    size = np.asarray(half_size, dtype=np.float64).reshape(-1)
    if size.size == 1:
        extent_z = float(size[0])
    elif size.size == 3:
        extent_z = float(size[2])
    else:
        raise ValueError(f"half_size must be a scalar or length-3, got {half_size!r}")

    delta = np.asarray(offset, dtype=np.float64).reshape(-1)
    if delta.size != 3:
        raise ValueError(f"offset must be length-3 (xyz), got {offset!r}")

    normalized = _normalize_flex_name(flex_name)
    _ = grid_size or infer_flex_grid_size(normalized)

    model = spec.compile()
    data = mj.MjData(model)
    if configure is not None:
        configure(model, data)
    mj.mj_forward(model, data)
    vertex_index, _taxel_id = flex_vertex_for_fk_grid(
        model, normalized, grid_row, grid_col
    )

    position, quat = flex_taxel_frame(model, data, normalized, vertex_index)
    rotation = np.empty(9, dtype=np.float64)
    mj.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    local = np.array(
        [0.0, 0.0, extent_z + float(clearance)], dtype=np.float64
    ) + delta
    centre = position + rotation @ local
    return np.asarray(centre, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def add_cube_on_taxel(
    spec: mj.MjSpec,
    *,
    flex_name: str,
    grid_row: int,
    grid_col: int,
    name: str | None = None,
    grid_size: tuple[int, int] | None = None,
    quat: Sequence[float] | None = None,
    half_size: float | Sequence[float] = DEFAULT_TAXEL_CUBE_HALF_SIZE,
    mass: float = DEFAULT_TAXEL_CUBE_MASS,
    rgba: Sequence[float] = DEFAULT_TAXEL_CUBE_RGBA,
    contype: int = DEFAULT_CUBE_CONTYPE,
    conaffinity: int = DEFAULT_CUBE_CONAFFINITY,
    freejoint: bool = True,
    joint_name: str | None = None,
    clearance: float = DEFAULT_TAXEL_CUBE_CLEARANCE,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    configure: Callable[[mj.MjModel, mj.MjData], None] | None = None,
) -> mj.MjsBody:
    """Spawn a small, heavy cube above one taxel on a regular-grid flex pad.

    Parameters
    ----------
    flex_name:
        Flex to target, e.g. ``\"uspa46_1\"`` or ``\"flex_uspa46_1\"``.
    grid_row, grid_col:
        Taxel coordinates on the pad grid. For ``uspa46`` pads this is a
        4×6 grid (4 rows, 6 columns), zero-indexed from one corner.
    grid_size:
        Optional ``(n_cols, n_rows)`` override when the flex name does not
        match a known pattern.
    quat:
        World orientation (wxyz). If ``None``, matches the taxel frame.
    half_size, mass:
        Defaults are a 2.5 mm cube with 2.0 kg mass (dense point load).
    clearance:
        Gap between the flex surface and the cube bottom along the taxel normal.
    offset:
        Extra taxel-local XYZ shift (same convention as sticky cubes).
    configure:
        Optional ``(model, data) -> None`` applied before reading taxel pose
        (e.g. set joint/actuator initials so spawn matches the demo pose).
    """
    normalized = _normalize_flex_name(flex_name)
    n_cols, n_rows = grid_size or infer_flex_grid_size(normalized)
    body_name = name or f"taxel_cube_{normalized}_r{grid_row}_c{grid_col}"
    pos, taxel_quat = spawn_pose_above_taxel(
        spec,
        flex_name=normalized,
        grid_row=grid_row,
        grid_col=grid_col,
        grid_size=(n_cols, n_rows),
        half_size=half_size,
        clearance=clearance,
        offset=offset,
        configure=configure,
    )
    return add_cube(
        spec,
        name=body_name,
        pos=pos,
        quat=taxel_quat if quat is None else quat,
        half_size=half_size,
        mass=mass,
        rgba=rgba,
        contype=contype,
        conaffinity=conaffinity,
        freejoint=freejoint,
        joint_name=joint_name,
        above_palm=False,
    )


def add_sticky_cube_on_taxel(
    spec: mj.MjSpec,
    *,
    flex_name: str,
    grid_row: int,
    grid_col: int,
    name: str | None = None,
    half_size: float | Sequence[float] = DEFAULT_STICKY_CUBE_HALF_SIZE,
    mass: float = DEFAULT_STICKY_CUBE_MASS,
    rgba: Sequence[float] = DEFAULT_STICKY_CUBE_RGBA,
    contype: int = DEFAULT_CUBE_CONTYPE,
    conaffinity: int = DEFAULT_CUBE_CONAFFINITY,
    clearance: float = DEFAULT_STICKY_CUBE_CLEARANCE,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> mj.MjsBody:
    """Attach a cube to one flex taxel so it stays stuck as the pad moves.

    The cube is a child of the taxel vertex body, offset along the local
    outward normal (+z). Intended for fingertip pads (``mf_tip``, ``if_tip``,
    …) but works for any flex that maps through ``flex_vertex_for_fk_grid``.

    Parameters
    ----------
    flex_name:
        Flex to target, e.g. ``\"mf_tip\"`` or ``\"flex_mf_tip\"``.
    grid_row, grid_col:
        FK panel cell (0-indexed). Fingertips use a sparse 6×6 canvas.
    half_size, mass:
        Defaults are a ~4.6 mm cube with 20 g mass.
    clearance:
        Extra gap along the taxel normal beyond ``half_size[2]`` (0 → just touching).
    offset:
        Extra shift in the taxel-local frame (x shear, y shear, z normal).
    """
    size = np.asarray(half_size, dtype=np.float64).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 3)
    elif size.size != 3:
        raise ValueError(f"half_size must be a scalar or length-3, got {half_size!r}")
    if np.any(size <= 0.0):
        raise ValueError(f"half_size must be > 0, got {half_size!r}")

    delta = np.asarray(offset, dtype=np.float64).reshape(-1)
    if delta.size != 3:
        raise ValueError(f"offset must be length-3 (xyz), got {offset!r}")

    normalized = _normalize_flex_name(flex_name)
    vertex_body_name, _vertex_index, taxel_id = taxel_vertex_body_name(
        spec,
        flex_name=normalized,
        grid_row=grid_row,
        grid_col=grid_col,
    )
    parent = spec.body(vertex_body_name)
    if parent is None:
        raise ValueError(f"Vertex body '{vertex_body_name}' not found in spec")

    body_name = name or (
        f"sticky_cube_{normalized}_r{grid_row}_c{grid_col}"
        if taxel_id < 0
        else f"sticky_cube_taxel_{taxel_id:03d}"
    )
    if spec.body(body_name) is not None:
        raise ValueError(f"Body '{body_name}' already exists in the spec")

    local_pos = np.array(
        [0.0, 0.0, float(size[2]) + float(clearance)], dtype=np.float64
    ) + delta

    body = parent.add_body(
        name=body_name,
        pos=local_pos.tolist(),
    )
    body.add_geom(
        name=body_name,
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size.tolist(),
        mass=float(mass),
        rgba=np.asarray(rgba, dtype=np.float64).tolist(),
        contype=int(contype),
        conaffinity=int(conaffinity),
        condim=3,
        friction=[0.8, 0.005, 0.0001],
        group=0,
    )
    return body


def spawn_pose_above_palm(
    spec: mj.MjSpec,
    *,
    half_size: float | Sequence[float] = DEFAULT_CUBE_HALF_SIZE,
    clearance: float = DEFAULT_CUBE_CLEARANCE,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Cube centre pose above the palm pads (xy over pads, z above the surface)."""
    return spawn_pose_above_flex(
        spec,
        flex_substring=PALM_FLEX_SUBSTRING,
        half_size=half_size,
        clearance=clearance,
        offset=offset,
    )


def _resolve_spawn_pos(
    spec: mj.MjSpec,
    *,
    pos: Sequence[float] | None,
    flex_name: str | None,
    above_palm: bool,
    half_size: float | Sequence[float],
    clearance: float,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray | Sequence[float]:
    delta = np.asarray(offset, dtype=np.float64).reshape(-1)
    if delta.size != 3:
        raise ValueError(f"offset must be length-3 (xyz), got {offset!r}")

    if pos is not None:
        return np.asarray(pos, dtype=np.float64) + delta
    if flex_name is not None:
        return spawn_pose_above_flex(
            spec,
            flex_name=flex_name,
            half_size=half_size,
            clearance=clearance,
            offset=delta,
        )
    if above_palm:
        return spawn_pose_above_palm(
            spec, half_size=half_size, clearance=clearance, offset=delta
        )
    raise ValueError(
        "pos must be provided when above_palm=False and flex_name is not set"
    )


def add_cube(
    spec: mj.MjSpec,
    *,
    name: str = DEFAULT_CUBE_NAME,
    pos: Sequence[float] | None = None,
    quat: Sequence[float] = DEFAULT_CUBE_QUAT,
    half_size: float | Sequence[float] = DEFAULT_CUBE_HALF_SIZE,
    mass: float = DEFAULT_CUBE_MASS,
    rgba: Sequence[float] = DEFAULT_CUBE_RGBA,
    contype: int = DEFAULT_CUBE_CONTYPE,
    conaffinity: int = DEFAULT_CUBE_CONAFFINITY,
    freejoint: bool = True,
    joint_name: str | None = None,
    above_palm: bool = True,
    flex_name: str | None = None,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    clearance: float = DEFAULT_CUBE_CLEARANCE,
) -> mj.MjsBody:
    """Add a free (or welded) cube to ``spec.worldbody``.

    Parameters
    ----------
    spec:
        Mutable MuJoCo model specification (load with ``MjSpec.from_file``).
    name:
        Body / collision-geom base name.
    pos:
        World position. If ``None``, spawn from ``flex_name`` or palm pads
        (when ``above_palm``); otherwise requires an explicit ``pos``.
    quat:
        World orientation quaternion (w, x, y, z).
    half_size:
        Box half-extents in metres. Scalar → cube; length-3 → rectangular box.
    mass:
        Cube mass in kg.
    rgba:
        Visual colour.
    contype, conaffinity:
        MuJoCo collision bitmasks. Defaults collide with the flex skins and floor.
    freejoint:
        If True, attach a free joint so the cube can fall / be teleoped.
    joint_name:
        Optional free-joint name (defaults to ``\"{name}_freejoint\"``).
    above_palm:
        When ``pos`` and ``flex_name`` are None, place the cube over the palm
        sensors.
    flex_name:
        Spawn above this flex (``flex_`` prefix optional), e.g. ``mf_tip`` or
        ``flex_uspa46_1``. Overrides ``above_palm``.
    offset:
        World-frame XYZ added to the spawn pose (flex/palm centre or ``pos``).
    clearance:
        Extra gap above the flex surface when auto-spawning.

    Returns
    -------
    body:
        The created ``MjsBody``.
    """
    if spec.body(name) is not None:
        raise ValueError(f"Body '{name}' already exists in the spec")

    size = np.asarray(half_size, dtype=np.float64).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 3)
    elif size.size != 3:
        raise ValueError(f"half_size must be a scalar or length-3, got {half_size!r}")

    pos = _resolve_spawn_pos(
        spec,
        pos=pos,
        flex_name=flex_name,
        above_palm=above_palm,
        half_size=size,
        clearance=clearance,
        offset=offset,
    )

    body = spec.worldbody.add_body(
        name=name,
        pos=np.asarray(pos, dtype=np.float64).tolist(),
        quat=np.asarray(quat, dtype=np.float64).tolist(),
    )
    if freejoint:
        joint = body.add_freejoint()
        joint.name = joint_name or f"{name}_freejoint"

    body.add_geom(
        name=name,
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size.tolist(),
        mass=float(mass),
        rgba=np.asarray(rgba, dtype=np.float64).tolist(),
        contype=int(contype),
        conaffinity=int(conaffinity),
        condim=3,
        friction=[0.8, 0.005, 0.0001],
        # Group 0 is visible in the default MuJoCo viewer; group 3 is hidden.
        group=0,
    )
    return body


def _tetris_cells(shape: str) -> tuple[tuple[int, int], ...]:
    key = shape.upper()
    if key not in TETRIS_SHAPES:
        raise ValueError(
            f"Unknown tetris shape {shape!r}; choose from {TETRIS_SHAPE_NAMES}"
        )
    return TETRIS_SHAPES[key]


def _euler_xyz_to_quat(euler: Sequence[float]) -> np.ndarray:
    """Convert XYZ intrinsic Euler angles (radians) to MuJoCo quat (w, x, y, z)."""
    roll, pitch, yaw = (float(a) for a in euler)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _resolve_tetris_quat(
    quat: Sequence[float] | None,
    euler: Sequence[float] | None,
) -> np.ndarray:
    if euler is not None and quat is not None:
        raise ValueError("Pass only one of quat or euler, not both")
    if euler is not None:
        e = np.asarray(euler, dtype=np.float64).reshape(-1)
        if e.size != 3:
            raise ValueError(f"euler must be length-3 (xyz rad), got {euler!r}")
        return _euler_xyz_to_quat(e)
    if quat is None:
        return np.asarray(DEFAULT_CUBE_QUAT, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"quat must be length-4 (wxyz), got {quat!r}")
    return q


def _tetris_local_block_positions(
    cells: Sequence[tuple[int, int]],
    block_half: float,
) -> np.ndarray:
    """Centre each unit square so the piece COM is near the origin (xy)."""
    pitch = 2.0 * float(block_half)
    pts = np.asarray(cells, dtype=np.float64) * pitch
    pts -= pts.mean(axis=0)
    # (n, 3): lay pieces in the xy plane; z is body-local height.
    return np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(cells))])


def add_tetris_part(
    spec: mj.MjSpec,
    *,
    shape: str = "T",
    name: str | None = None,
    pos: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
    euler: Sequence[float] | None = None,
    scale: float = 1.0,
    block_half: float = DEFAULT_TETRIS_BLOCK_HALF,
    block_mass: float = DEFAULT_TETRIS_BLOCK_MASS,
    rgba: Sequence[float] | None = None,
    contype: int = DEFAULT_CUBE_CONTYPE,
    conaffinity: int = DEFAULT_CUBE_CONAFFINITY,
    freejoint: bool = True,
    joint_name: str | None = None,
    above_palm: bool = True,
    flex_name: str | None = None,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    clearance: float = DEFAULT_TETRIS_CLEARANCE,
) -> mj.MjsBody:
    """Add one rigid tetromino (4 box geoms) to ``spec.worldbody``.

    Parameters
    ----------
    scale:
        Uniform size multiplier applied to ``block_half`` (mass scales as
        ``scale**3`` to keep density roughly constant).
    euler:
        Optional XYZ Euler rotation in radians. Mutually exclusive with a
        non-default ``quat``.
    quat:
        World orientation quaternion (w, x, y, z). Default identity.
    flex_name:
        Spawn above this flex (``flex_`` prefix optional). Overrides
        ``above_palm`` when ``pos`` is None.
    offset:
        World-frame XYZ added to the spawn pose (flex/palm centre or ``pos``).
    """
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")

    cells = _tetris_cells(shape)
    shape_key = shape.upper()
    body_name = name or f"tetris_{shape_key}"
    if spec.body(body_name) is not None:
        raise ValueError(f"Body '{body_name}' already exists in the spec")

    half = float(block_half) * float(scale)
    mass = float(block_mass) * (float(scale) ** 3)
    orientation = _resolve_tetris_quat(quat, euler)

    pos = _resolve_spawn_pos(
        spec,
        pos=pos,
        flex_name=flex_name,
        above_palm=above_palm,
        half_size=(half, half, half),
        clearance=clearance,
        offset=offset,
    )

    color = (
        np.asarray(rgba, dtype=np.float64)
        if rgba is not None
        else np.asarray(TETRIS_COLORS[shape_key], dtype=np.float64)
    )
    local = _tetris_local_block_positions(cells, half)

    body = spec.worldbody.add_body(
        name=body_name,
        pos=np.asarray(pos, dtype=np.float64).tolist(),
        quat=orientation.tolist(),
    )
    if freejoint:
        joint = body.add_freejoint()
        joint.name = joint_name or f"{body_name}_freejoint"

    size = [half, half, half]
    for i, offset in enumerate(local):
        body.add_geom(
            name=f"{body_name}_block{i}",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=offset.tolist(),
            size=size,
            mass=mass,
            rgba=color.tolist(),
            contype=int(contype),
            conaffinity=int(conaffinity),
            condim=3,
            friction=[0.8, 0.005, 0.0001],
            group=0,
        )
    return body


def add_tetris_parts(
    spec: mj.MjSpec,
    n: int,
    *,
    shapes: Sequence[str] | None = None,
    name_prefix: str = "tetris",
    pos: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
    euler: Sequence[float] | None = None,
    scale: float = 1.0,
    block_half: float = DEFAULT_TETRIS_BLOCK_HALF,
    block_mass: float = DEFAULT_TETRIS_BLOCK_MASS,
    contype: int = DEFAULT_CUBE_CONTYPE,
    conaffinity: int = DEFAULT_CUBE_CONAFFINITY,
    freejoint: bool = True,
    above_palm: bool = True,
    flex_name: str | None = None,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    clearance: float = DEFAULT_TETRIS_CLEARANCE,
    spacing: float = DEFAULT_TETRIS_SPACING,
    seed: int | None = None,
) -> list[mj.MjsBody]:
    """Add ``n`` free tetrominoes to ``spec.worldbody``.

    Parameters
    ----------
    spec:
        Mutable MuJoCo model specification.
    n:
        Number of pieces to add (must be >= 0).
    shapes:
        Optional sequence of shape letters (``I/O/T/L/J/S/Z``). Cycled if
        shorter than ``n``. If ``None``, shapes are chosen uniformly at random.
    name_prefix:
        Body names become ``{name_prefix}_{i}_{shape}``.
    pos:
        World position of the first piece. If ``None``, spawn from ``flex_name``
        or palm pads; remaining pieces are offset along +x.
    quat, euler:
        Orientation for every piece (see ``add_tetris_part``).
    scale:
        Uniform size multiplier for every piece.
    flex_name:
        Spawn the first piece above this flex (``flex_`` prefix optional).
    offset:
        World-frame XYZ added to the first spawn pose.
    spacing:
        Centre-to-centre gap along +x between consecutive pieces (also scaled).
    seed:
        RNG seed used when ``shapes`` is ``None``.

    Returns
    -------
    bodies:
        The created ``MjsBody`` list (length ``n``).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return []
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")

    rng = np.random.default_rng(seed)
    if shapes is None:
        chosen = [str(rng.choice(TETRIS_SHAPE_NAMES)) for _ in range(n)]
    else:
        if not shapes:
            raise ValueError("shapes must be a non-empty sequence when provided")
        chosen = [shapes[i % len(shapes)].upper() for i in range(n)]
        for s in chosen:
            _tetris_cells(s)

    half = float(block_half) * float(scale)
    gap = float(spacing) * float(scale)
    base = np.asarray(
        _resolve_spawn_pos(
            spec,
            pos=pos,
            flex_name=flex_name,
            above_palm=above_palm,
            half_size=(half, half, half),
            clearance=clearance,
            offset=offset,
        ),
        dtype=np.float64,
    )

    bodies: list[mj.MjsBody] = []
    for i, shape in enumerate(chosen):
        piece_pos = base + np.array([i * gap, 0.0, 0.0], dtype=np.float64)
        body = add_tetris_part(
            spec,
            shape=shape,
            name=f"{name_prefix}_{i}_{shape}",
            pos=piece_pos,
            quat=quat,
            euler=euler,
            scale=scale,
            block_half=block_half,
            block_mass=block_mass,
            contype=contype,
            conaffinity=conaffinity,
            freejoint=freejoint,
            above_palm=False,
            clearance=clearance,
        )
        bodies.append(body)
    return bodies
