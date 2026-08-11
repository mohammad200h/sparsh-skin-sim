"""Leap+XELA taxel forward kinematics and Open3D visualization (non-ROS).

Port of ``leapXela_taxels_forewardkinematic/fk_taxels_demo.py`` from the ROS
workspace. Joint angles and per-taxel forces are passed in directly instead of
via ROS topics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import einops
import mujoco as mj
import numpy as np
import pytorch_kinematics as pk
import torch

from flex_util import flex_vertex_body_ids, list_flex_names

N_TAXELS = 368

# Patch link frames in hand_ss.urdf (same taxel counts as Allegro XELA flatten order).
XELA_FLATTEN_ORDER = {
    "3aftc_palm_link": 30,
    "link_15_4x4_palm_link": 16,
    "link_14_4x4_palm_link": 16,
    "0aftc_palm_link": 30,
    "link_2_4x4_palm_link": 16,
    "link_1A_4x4_palm_link": 16,
    "1B_4x4_palm_link": 16,
    "1aftc_palm_link": 30,
    "link_6_4x4_palm_link": 16,
    "link_5A_4x4_palm_link": 16,
    "5B_4x4_palm_link": 16,
    "2aftc_palm_link": 30,
    "link_10_4x4_palm_link": 16,
    "link_9A_4x4_palm_link": 16,
    "9B_4x4_palm_link": 16,
    "ahr_palm_2_4x6_palm_link": 24,
    "ahr_palm_1_4x6_palm_link": 24,
    "ahr_palm_3_4x6_palm_link": 24,
}

_PATCH_TO_TAXEL_MAP = {
    "3aftc_palm_link": ("TH", "tip", True, 0, False),
    "link_15_4x4_palm_link": ("TH", "second", False, 4, False),
    "link_14_4x4_palm_link": ("TH", "third", False, 4, False),
    "0aftc_palm_link": ("IF", "tip", True, 0, False),
    "link_2_4x4_palm_link": ("IF", "second", False, 4, False),
    "link_1A_4x4_palm_link": ("IF", "third", False, 4, False),
    "1B_4x4_palm_link": ("IF", "fourth", False, 4, False),
    "1aftc_palm_link": ("MF", "tip", True, 0, False),
    "link_6_4x4_palm_link": ("MF", "second", False, 4, False),
    "link_5A_4x4_palm_link": ("MF", "third", False, 4, False),
    "5B_4x4_palm_link": ("MF", "fourth", False, 4, False),
    "2aftc_palm_link": ("RF", "tip", True, 0, False),
    "link_10_4x4_palm_link": ("RF", "second", False, 4, False),
    "link_9A_4x4_palm_link": ("RF", "third", False, 4, False),
    "9B_4x4_palm_link": ("RF", "fourth", False, 4, False),
    "ahr_palm_2_4x6_palm_link": ("Palm", "up_right", False, 4, True),
    "ahr_palm_1_4x6_palm_link": ("Palm", "up_left", False, 4, True),
    "ahr_palm_3_4x6_palm_link": ("Palm", "down_left", False, 4, True),
}

# Palm flexes in the MuJoCo scene and their JSON patch keys.
FLEX_TO_PALM_PATCH: dict[str, str] = {
    "flex_uspa46_1": "up_left",
    "flex_uspa46_2": "up_right",
    "flex_uspa46_3": "down_left",
}
# URDF links whose local taxel grid is mirrored vs the MuJoCo flex mesh so that
# FK left/right matches the simulated pad (and hardware taxel placement).
FK_FLIP_IN_PLANE_LINKS = frozenset({"ahr_palm_2_4x6_palm_link"})
# Finger 4x4 pads: FK row axis is mirrored across the finger relative to MuJoCo
# (IF/RF sides swapped). Flip local ``yy`` only — see get_fk_taxel_frames.
FK_FLIP_Y_LINKS = frozenset(
    name for name in XELA_FLATTEN_ORDER if "4x4" in name
)
# MuJoCo flex name -> (finger, JSON patch, is_tip).  ``px``/``md`` geom names are
# swapped vs JSON ``second``/``third`` on IF/MF/RF; map by actual taxel ownership.
FLEX_TO_FINGER_PATCH: dict[str, tuple[str, str, bool]] = {
    "flex_if_bs_uspa44": ("IF", "fourth", False),
    "flex_if_px_uspa44": ("IF", "third", False),
    "flex_if_md_uspa44": ("IF", "second", False),
    "flex_if_tip": ("IF", "tip", True),
    "flex_mf_bs_uspa44": ("MF", "fourth", False),
    "flex_mf_px_uspa44": ("MF", "third", False),
    "flex_mf_md_uspa44": ("MF", "second", False),
    "flex_mf_tip": ("MF", "tip", True),
    "flex_rf_bs_uspa44": ("RF", "fourth", False),
    "flex_rf_px_uspa44": ("RF", "third", False),
    "flex_rf_md_uspa44": ("RF", "second", False),
    "flex_rf_tip": ("RF", "tip", True),
    "flex_th_px_uspa44": ("TH", "third", False),
    "flex_th_ds_uspa44": ("TH", "second", False),
    "flex_th_tip": ("TH", "tip", True),
}

LEAP_JOINT_ORDER = [
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
]

_MUJOCO_PALM_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
_MUJOCO_PALM_T = np.array([0.0, 0.0, 0.1], dtype=np.float64)

_DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parent / "leapXELA_urdf" / "hand_ss.urdf"
)
_DEFAULT_TAXEL_MAP_PATH = (
    Path(__file__).resolve().parent
    / "leapXELA_model"
    / "leapxela"
    / "leap_sensor_taxel_map.json"
)

_kinematic_chain = None
_cached_urdf_path: Path | None = None
_taxel_ids_in_fk_order: np.ndarray | None = None
_flex_taxel_mapper: dict[int, tuple[str, int]] | None = None
_flex_taxel_mapper_model_id: int | None = None


@dataclass(frozen=True)
class FkTaxelResult:
    """FK output for one timestep."""

    positions: np.ndarray
    rotations: np.ndarray
    forces_local: np.ndarray | None
    forces_world: np.ndarray | None
    positions_deformed: np.ndarray | None


def _default_urdf_path() -> Path:
    if _DEFAULT_URDF_PATH.is_file():
        return _DEFAULT_URDF_PATH
    raise FileNotFoundError(f"URDF not found: {_DEFAULT_URDF_PATH}")


def _default_taxel_map_path() -> Path:
    if _DEFAULT_TAXEL_MAP_PATH.is_file():
        return _DEFAULT_TAXEL_MAP_PATH
    raise FileNotFoundError(
        f"Could not locate leap_sensor_taxel_map.json at {_DEFAULT_TAXEL_MAP_PATH}"
    )


def get_sensor_grid(patch_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    if "aftc" in patch_name:
        h, w = 0.031, 0.039
        h_res, w_res = 6, 6
        x = np.linspace(0.5 - h_res / 2, h_res / 2 + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx_, yy_ = np.meshgrid(x, y)
        xx = np.concatenate([xx_[:4, :].flatten(), xx_[-2, 1:-1], xx_[-1, 2:-2]], axis=0)
        yy = np.concatenate([yy_[:4, :].flatten(), yy_[-2, 1:-1], yy_[-1, 2:-2]], axis=0)
        d = 0.029
    elif "4x4" in patch_name:
        h, w = 0.026, 0.024
        h_res, w_res = 4, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
        d = 0.0044
    elif "4x6" in patch_name:
        offset_x = 0.00435
        offset_y = 0.00425
        x_dist = 0.00725
        y_dist = 0.00717
        d = 0.0
        n_cols, n_rows = 6, 4
        x = offset_x + np.arange(n_cols) * x_dist
        y = offset_y + np.arange(n_rows) * y_dist
        xx, yy = np.meshgrid(x, y)
    else:
        raise ValueError(f"Unknown patch name: {patch_name}")
    return xx, yy, d


def _tip_ids_in_fk_grid_order(ids: list[int]) -> list[int]:
    if len(ids) != 30:
        raise ValueError(f"tip expected 30 ids, got {len(ids)}")
    grid = np.full((6, 6), -1, dtype=np.int32)
    grid[0, 2:6] = ids[0:4]
    grid[1, 1:6] = ids[4:9]
    grid[2, 0:6] = ids[9:15]
    grid[3, 0:6] = ids[15:21]
    grid[4, 1:6] = ids[21:26]
    grid[5, 2:6] = ids[26:30]
    out = np.full((6, 6), -1, dtype=np.int32)
    for r in range(6):
        for c in range(6):
            out[r, c] = grid[5 - c, 5 - r]
    return out[:4, :].reshape(-1).tolist() + out[4, 1:5].tolist() + out[5, 2:4].tolist()


def _palm_ids_in_fk_grid_order(ids: list[int], patch: str) -> list[int]:
    if len(ids) != 24:
        raise ValueError(f"palm expected 24 ids, got {len(ids)}")
    hw = np.asarray(ids, dtype=np.int32).reshape(6, 4)
    if patch == "up_right":
        out = hw.T
    else:
        out = np.flipud(np.rot90(hw, -1))
    return out.reshape(-1).tolist()


def flex_grid_flip_in_plane(flex_name: str) -> bool:
    """True when the MuJoCo flex mesh is 180° in-plane vs the FK URDF patch."""
    normalized = flex_name if flex_name.startswith("flex_") else f"flex_{flex_name}"
    return normalized == "flex_uspa46_2"


def fk_palm_taxel_id(
    patch: str,
    grid_row: int,
    grid_col: int,
    map_path: Path | None = None,
) -> int:
    """Hardware taxel id for a palm patch FK grid cell (4 rows × 6 columns)."""
    map_path = map_path or _default_taxel_map_path()
    with map_path.open(encoding="utf-8") as f:
        map_dict = json.load(f)
    ids = _flatten_patch_taxel_ids(map_dict, "Palm", patch, False, is_palm=True)
    grid = np.asarray(ids, dtype=np.int32).reshape(4, 6)
    if not (0 <= grid_row < 4 and 0 <= grid_col < 6):
        raise ValueError(
            f"Palm grid cell ({grid_row}, {grid_col}) out of range for 4×6 patch"
        )
    return int(grid[grid_row, grid_col])


def _tip_fk_canvas(
    finger: str,
    map_path: Path | None = None,
) -> np.ndarray:
    """Return the 6×6 FK tip panel (-1 = empty cell)."""
    map_path = map_path or _default_taxel_map_path()
    with map_path.open(encoding="utf-8") as f:
        map_dict = json.load(f)
    ids = _flatten_patch_taxel_ids(map_dict, finger, "tip", True, is_palm=False)
    grid = np.full((6, 6), -1, dtype=np.int32)
    grid[0, 2:6] = ids[0:4]
    grid[1, 1:6] = ids[4:9]
    grid[2, 0:6] = ids[9:15]
    grid[3, 0:6] = ids[15:21]
    grid[4, 1:6] = ids[21:26]
    grid[5, 2:6] = ids[26:30]
    out = np.full((6, 6), -1, dtype=np.int32)
    for row in range(6):
        for col in range(6):
            out[row, col] = grid[5 - col, 5 - row]
    return out


def fk_tip_taxel_id(
    finger: str,
    grid_row: int,
    grid_col: int,
    map_path: Path | None = None,
) -> int:
    """Hardware taxel id for a fingertip FK panel cell (6×6 canvas, sparse)."""
    canvas = _tip_fk_canvas(finger, map_path=map_path)
    if not (0 <= grid_row < 6 and 0 <= grid_col < 6):
        raise ValueError(
            f"Tip grid cell ({grid_row}, {grid_col}) out of range for 6×6 canvas"
        )
    taxel_id = int(canvas[grid_row, grid_col])
    if taxel_id < 0:
        raise ValueError(
            f"No taxel at fingertip grid ({grid_row}, {grid_col}) for {finger}"
        )
    return taxel_id


def fk_finger_taxel_id(
    finger: str,
    patch: str,
    grid_row: int,
    grid_col: int,
    map_path: Path | None = None,
) -> int:
    """Hardware taxel id for a 4×4 finger-segment FK grid cell."""
    map_path = map_path or _default_taxel_map_path()
    with map_path.open(encoding="utf-8") as f:
        map_dict = json.load(f)
    ids = _flatten_patch_taxel_ids(map_dict, finger, patch, False, is_palm=False)
    grid = np.asarray(ids, dtype=np.int32).reshape(4, 4)
    if not (0 <= grid_row < 4 and 0 <= grid_col < 4):
        raise ValueError(
            f"Finger grid cell ({grid_row}, {grid_col}) out of range for 4×4 patch"
        )
    return int(grid[grid_row, grid_col])


def flex_vertex_for_fk_grid(
    model: mj.MjModel,
    flex_name: str,
    grid_row: int,
    grid_col: int,
) -> tuple[int, int]:
    """Map FK panel ``(row, col)`` on a flex pad to ``(vertex_index, taxel_id)``."""
    normalized = flex_name if flex_name.startswith("flex_") else f"flex_{flex_name}"
    patch = FLEX_TO_PALM_PATCH.get(normalized)
    finger_info = FLEX_TO_FINGER_PATCH.get(normalized)

    if patch is not None:
        taxel_id = fk_palm_taxel_id(patch, grid_row, grid_col)
    elif finger_info is not None:
        finger, json_patch, is_tip = finger_info
        if is_tip:
            taxel_id = fk_tip_taxel_id(finger, grid_row, grid_col)
        else:
            taxel_id = fk_finger_taxel_id(finger, json_patch, grid_row, grid_col)
    else:
        n_cols, n_rows = (6, 4) if "uspa46" in normalized else (4, 4)
        vertex = int(grid_col * n_rows + grid_row)
        return vertex, -1

    mapper = build_flex_taxel_mapper(model)
    flex_for_taxel, vertex = mapper[taxel_id]
    if flex_for_taxel != normalized:
        raise ValueError(
            f"Taxel {taxel_id} at grid ({grid_row}, {grid_col}) "
            f"belongs to '{flex_for_taxel}', not '{normalized}'"
        )
    return int(vertex), int(taxel_id)


def _flatten_patch_taxel_ids(
    map_dict: dict,
    finger: str,
    patch: str,
    is_tip: bool,
    is_palm: bool = False,
) -> list[int]:
    patch_dict = map_dict[finger][patch]
    row_keys = sorted(patch_dict.keys(), key=lambda k: int(k))
    ids: list[int] = []
    for key in row_keys:
        ids.extend(int(v) for v in patch_dict[key])
    if is_tip:
        return _tip_ids_in_fk_grid_order(ids)
    if is_palm:
        return _palm_ids_in_fk_grid_order(ids, patch)
    return ids


def build_taxel_ids_in_fk_order(map_path: Path | None = None) -> np.ndarray:
    map_path = map_path or _default_taxel_map_path()
    with map_path.open(encoding="utf-8") as f:
        map_dict = json.load(f)

    taxel_ids: list[int] = []
    for link_name, num_sensors in XELA_FLATTEN_ORDER.items():
        finger, patch, is_tip, _width, is_palm = _PATCH_TO_TAXEL_MAP[link_name]
        ids = _flatten_patch_taxel_ids(map_dict, finger, patch, is_tip, is_palm=is_palm)
        if len(ids) != num_sensors:
            raise ValueError(
                f"{link_name} ({finger}/{patch}): expected {num_sensors} ids, got {len(ids)}"
            )
        taxel_ids.extend(ids)

    out = np.asarray(taxel_ids, dtype=np.int32)
    if out.shape[0] != N_TAXELS:
        raise ValueError(f"Expected {N_TAXELS} taxel ids, got {out.shape[0]}")
    if not np.array_equal(np.sort(out), np.arange(N_TAXELS)):
        raise ValueError("Taxel ids from map do not cover 0..367 exactly once")
    return out


def taxel_ids_in_fk_order() -> np.ndarray:
    global _taxel_ids_in_fk_order
    if _taxel_ids_in_fk_order is None:
        _taxel_ids_in_fk_order = build_taxel_ids_in_fk_order()
    return _taxel_ids_in_fk_order


def urdf_base_to_mujoco_world(
    positions: np.ndarray, rotations: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    squeeze = positions.ndim == 2
    if squeeze:
        positions = positions[None, ...]
    pos_w = np.einsum("ij,tsj->tsi", _MUJOCO_PALM_R, positions) + _MUJOCO_PALM_T
    if rotations is None:
        return pos_w[0] if squeeze else pos_w
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim == 3:
        rotations = rotations[None, ...]
    rot_w = np.einsum("ij,tsjk->tsik", _MUJOCO_PALM_R, rotations)
    if squeeze:
        return pos_w[0], rot_w[0]
    return pos_w, rot_w


def _get_kinematic_chain(urdf_path: Path | None = None):
    global _kinematic_chain, _cached_urdf_path
    urdf_path = Path(urdf_path) if urdf_path is not None else _default_urdf_path()
    urdf_path = urdf_path.resolve()
    if _kinematic_chain is None or _cached_urdf_path != urdf_path:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found at {urdf_path}")
        urdf_text = urdf_path.read_text().lstrip()
        _kinematic_chain = pk.build_chain_from_urdf(urdf_text)
        _cached_urdf_path = urdf_path
    return _kinematic_chain


def get_fk_taxel_frames(
    joint_angles: np.ndarray,
    urdf_path: Path | None = None,
    mujoco_world: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    if joint_angles.ndim == 1:
        joint_angles = joint_angles[None, :]
    if joint_angles.ndim != 2 or joint_angles.shape[-1] != 16:
        raise ValueError(
            f"Expected joint_angles of shape (T, 16) or (16,), got {joint_angles.shape}"
        )

    kinematic_chain = _get_kinematic_chain(urdf_path)
    joint_angles_t = torch.tensor(joint_angles).float()
    joint_dict = {name: joint_angles_t[:, i] for i, name in enumerate(LEAP_JOINT_ORDER)}
    joint_poses = kinematic_chain.forward_kinematics(joint_dict)

    positions = []
    rotations = []
    for link_name, num_sensors in XELA_FLATTEN_ORDER.items():
        joint_pose = joint_poses[link_name].get_matrix().numpy()
        xx, yy, d = get_sensor_grid(link_name)
        if link_name in FK_FLIP_IN_PLANE_LINKS:
            # uspa46_2 is mounted 180° in-plane; mirror the patch frame so FK
            # taxel layout matches the MuJoCo flex mesh (flip_in_plane).
            xx = xx.max() + xx.min() - xx
            yy = yy.max() + yy.min() - yy
        elif link_name in FK_FLIP_Y_LINKS:
            # Finger 4x4: reverse across-finger local axis so IF/RF sides match
            # MuJoCo (otherwise FK lights the mirrored left/right corner).
            yy = yy.max() + yy.min() - yy
        sensor_local = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        sensor_local = np.concatenate([sensor_local, np.zeros_like(sensor_local)], axis=-1)
        sensor_local[..., -2] = d
        sensor_local[..., -1] = 1

        t = joint_pose.shape[0]
        pose_rep = einops.repeat(joint_pose, "t i j -> t s i j", s=num_sensors)
        pose_flat = einops.rearrange(pose_rep, "t s i j -> (t s) i j")
        local_flat = einops.repeat(sensor_local, "s c -> (t s) c", t=t)
        world_h = np.einsum("m i j, m j -> m i", pose_flat, local_flat)
        pose_flat[..., :, 3] = world_h
        pose_ts = einops.rearrange(pose_flat, "(t s) i j -> t s i j", s=num_sensors)
        positions.append(pose_ts[..., :3, 3])
        rotations.append(pose_ts[..., :3, :3])

    positions = np.concatenate(positions, axis=1)
    rotations = np.concatenate(rotations, axis=1)
    if mujoco_world:
        positions, rotations = urdf_base_to_mujoco_world(positions, rotations)
    return positions, rotations


def world_forces_to_local(
    forces_world: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    forces_world = np.asarray(forces_world, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    return np.einsum("nji,nj->ni", rotations, forces_world)


def forces_by_taxel_id_to_fk_order(forces_by_id: np.ndarray) -> np.ndarray:
    forces_by_id = np.asarray(forces_by_id, dtype=np.float64)
    if forces_by_id.shape != (N_TAXELS, 3):
        raise ValueError(f"Expected forces shape ({N_TAXELS}, 3), got {forces_by_id.shape}")
    return forces_by_id[taxel_ids_in_fk_order()]


def deform_taxel_positions(
    positions: np.ndarray,
    rotations: np.ndarray,
    forces_local: np.ndarray,
    scale: float = 0.02,
    max_disp: float = 0.015,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    forces_local = np.asarray(forces_local, dtype=np.float64)
    squeeze = positions.ndim == 2
    if squeeze:
        positions = positions[None, ...]
        rotations = rotations[None, ...]
        forces_local = forces_local[None, ...]

    world_disp = np.einsum("tsij,tsj->tsi", rotations, forces_local) * scale
    norms = np.linalg.norm(world_disp, axis=-1, keepdims=True)
    world_disp = np.where(
        norms > max_disp,
        world_disp * (max_disp / (norms + 1e-12)),
        world_disp,
    )
    deformed = positions + world_disp
    return deformed[0] if squeeze else deformed


def read_leap_joint_angles(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """Read the 16 Leap hand hinge joints in ``LEAP_JOINT_ORDER``."""
    angles = np.zeros(16, dtype=np.float32)
    for i, name in enumerate(LEAP_JOINT_ORDER):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model")
        qadr = int(model.jnt_qposadr[jid])
        angles[i] = float(data.qpos[qadr])
    return angles


def _taxel_site_on_body(model: mj.MjModel, body_id: int) -> int | None:
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, sid)
        if name and name.startswith("taxel_"):
            return int(name.split("_", 1)[1])
    return None


def build_flex_taxel_mapper(model: mj.MjModel) -> dict[int, tuple[str, int]]:
    """Map hardware taxel id -> (flex_name, vertex_index)."""
    mapper: dict[int, tuple[str, int]] = {}
    for flex_name in list_flex_names(model):
        body_ids = flex_vertex_body_ids(model, flex_name)
        for vertex_idx, body_id in enumerate(body_ids):
            taxel_id = _taxel_site_on_body(model, int(body_id))
            if taxel_id is None:
                continue
            mapper[int(taxel_id)] = (flex_name, int(vertex_idx))
    return mapper


def flex_forces_to_taxel_forces(
    model: mj.MjModel,
    flex_forces: dict[str, np.ndarray],
) -> np.ndarray:
    """Pack flex Kelvin-Voigt forces into a (368, 3) array keyed by taxel id."""
    global _flex_taxel_mapper, _flex_taxel_mapper_model_id
    model_id = id(model)
    if _flex_taxel_mapper is None or _flex_taxel_mapper_model_id != model_id:
        _flex_taxel_mapper = build_flex_taxel_mapper(model)
        _flex_taxel_mapper_model_id = model_id

    out = np.zeros((N_TAXELS, 3), dtype=np.float64)
    for taxel_id, (flex_name, vertex_idx) in _flex_taxel_mapper.items():
        force = flex_forces.get(flex_name)
        if force is None or vertex_idx >= force.shape[0]:
            continue
        out[taxel_id] = force[vertex_idx]
    return out


def compute_fk_taxels(
    joint_angles: np.ndarray,
    forces_by_taxel_id: np.ndarray | None = None,
    *,
    urdf_path: Path | None = None,
    deform_scale: float = 0.02,
    max_disp: float = 0.015,
    mujoco_world: bool = True,
) -> FkTaxelResult:
    """Run FK for 16 Leap joints and optional per-taxel forces.

    Parameters
    ----------
    joint_angles
        Shape ``(16,)`` or ``(T, 16)`` in ``LEAP_JOINT_ORDER``.
    forces_by_taxel_id
        Optional shape ``(368, 3)`` or ``(T, 368, 3)`` forces indexed by
        hardware taxel id (0..367), e.g. from ``flex_forces_to_taxel_forces``.
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    single = joint_angles.ndim == 1
    if single:
        joint_angles = joint_angles[None, :]

    positions, rotations = get_fk_taxel_frames(
        joint_angles, urdf_path=urdf_path, mujoco_world=mujoco_world
    )
    pos = positions[0]
    rot = rotations[0]

    forces_local = None
    forces_world = None
    pos_deformed = None
    if forces_by_taxel_id is not None:
        forces_by_taxel_id = np.asarray(forces_by_taxel_id, dtype=np.float64)
        if forces_by_taxel_id.ndim == 2:
            forces_by_taxel_id = forces_by_taxel_id[None, ...]
        forces_fk = np.stack(
            [forces_by_taxel_id_to_fk_order(f) for f in forces_by_taxel_id], axis=0
        )
        forces_local = forces_fk[0]
        forces_world = np.einsum("nij,nj->ni", rot, forces_local)
        pos_deformed = deform_taxel_positions(
            pos, rot, forces_local, scale=deform_scale, max_disp=max_disp
        )

    if single:
        return FkTaxelResult(pos, rot, forces_local, forces_world, pos_deformed)
    raise NotImplementedError("Batch FK results are not yet exposed; pass a (16,) joint vector")


def _taxel_patch_colors(num_taxels: int = N_TAXELS) -> np.ndarray:
    colors = np.zeros((num_taxels, 3), dtype=np.float64)
    n_patches = len(XELA_FLATTEN_ORDER)
    idx = 0
    for i, num_sensors in enumerate(XELA_FLATTEN_ORDER.values()):
        hue = i / n_patches
        h6 = hue * 6.0
        c = 0.95 * 0.75
        x = c * (1.0 - abs(h6 % 2.0 - 1.0))
        m = 0.95 - c
        if h6 < 1:
            rgb = (c, x, 0.0)
        elif h6 < 2:
            rgb = (x, c, 0.0)
        elif h6 < 3:
            rgb = (0.0, c, x)
        elif h6 < 4:
            rgb = (0.0, x, c)
        elif h6 < 5:
            rgb = (x, 0.0, c)
        else:
            rgb = (c, 0.0, x)
        colors[idx : idx + num_sensors] = np.array(rgb) + m
        idx += num_sensors
    return colors


def _force_magnitude_colors(forces_xyz: np.ndarray, vmax: float | None = None) -> np.ndarray:
    mag = np.linalg.norm(forces_xyz, axis=-1)
    if vmax is None:
        vmax = float(np.percentile(mag, 98)) if mag.size else 1.0
    vmax = max(vmax, 1e-6)
    t = np.clip(mag / vmax, 0.0, 1.0)
    colors = np.zeros((mag.shape[0], 3), dtype=np.float64)
    colors[:, 0] = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    colors[:, 1] = np.clip(1.0 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
    colors[:, 2] = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
    return colors


def _rotation_aligning_z_to(direction: np.ndarray) -> np.ndarray:
    v = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3)
    v = v / n
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    x, y, zc = axis
    k = np.array([[0.0, -zc, y], [zc, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


_SPHERE_TEMPLATE = None


def _unit_sphere_template(resolution: int = 6):
    global _SPHERE_TEMPLATE
    import open3d as o3d

    if _SPHERE_TEMPLATE is None or _SPHERE_TEMPLATE.get("resolution") != resolution:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        _SPHERE_TEMPLATE = {
            "resolution": resolution,
            "vertices": np.asarray(mesh.vertices, dtype=np.float64),
            "triangles": np.asarray(mesh.triangles, dtype=np.int32),
        }
    return _SPHERE_TEMPLATE


def _make_taxel_spheres(
    positions: np.ndarray,
    colors: np.ndarray,
    radius: float = 0.002,
    resolution: int = 6,
):
    import open3d as o3d

    positions = np.asarray(positions, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    tmpl = _unit_sphere_template(resolution)
    v0 = tmpl["vertices"]
    t0 = tmpl["triangles"]
    m = v0.shape[0]
    n = positions.shape[0]

    vertices = (v0[None, :, :] * radius + positions[:, None, :]).reshape(-1, 3)
    triangles = (t0[None, :, :] + (np.arange(n, dtype=np.int32) * m)[:, None, None]).reshape(-1, 3)
    vertex_colors = np.repeat(colors, m, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


def _make_force_arrows(
    positions: np.ndarray,
    forces_world: np.ndarray,
    arrow_scale: float = 0.08,
    max_len: float = 0.045,
    min_mag: float = 0.01,
    cylinder_radius: float = 0.0012,
    cone_radius: float = 0.0024,
):
    import open3d as o3d

    positions = np.asarray(positions, dtype=np.float64)
    forces_world = np.asarray(forces_world, dtype=np.float64)
    mag = np.linalg.norm(forces_world, axis=-1)
    keep = mag >= min_mag
    mesh = o3d.geometry.TriangleMesh()
    if not np.any(keep):
        return mesh

    origins = positions[keep]
    vecs = forces_world[keep] * arrow_scale
    lengths = np.linalg.norm(vecs, axis=-1)
    scale = np.ones_like(lengths)
    too_long = lengths > max_len
    scale[too_long] = max_len / (lengths[too_long] + 1e-12)
    vecs = vecs * scale[:, None]
    lengths = np.linalg.norm(vecs, axis=-1)
    cols = _force_magnitude_colors(forces_world[keep])

    for origin, vec, length, color in zip(origins, vecs, lengths, cols):
        if length < 1e-8:
            continue
        cone_h = min(0.35 * length, 0.012)
        cyl_h = max(length - cone_h, length * 0.5)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
            cylinder_height=cyl_h,
            cone_height=cone_h,
            resolution=12,
            cylinder_split=1,
            cone_split=1,
        )
        arrow.rotate(_rotation_aligning_z_to(vec), center=np.zeros(3))
        arrow.translate(origin)
        arrow.paint_uniform_color(color.tolist())
        mesh += arrow

    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
    return mesh


class FkTaxelVisualizer:
    """Non-blocking Open3D viewer updated from the MuJoCo demo loop."""

    def __init__(
        self,
        *,
        sphere_radius: float = 0.001,
        deform_scale: float = 0.02,
        max_disp: float = 0.015,
        color_by_force: bool = True,
        show_force_vectors: bool = True,
        arrow_scale: float = 0.08,
        arrow_max_len: float = 0.045,
        arrow_min_mag: float = 0.01,
        arrow_cylinder_radius: float = 0.0012,
        arrow_cone_radius: float = 0.0024,
    ) -> None:
        import open3d as o3d

        self.sphere_radius = sphere_radius
        self.deform_scale = deform_scale
        self.max_disp = max_disp
        self.color_by_force = color_by_force
        self.show_force_vectors = show_force_vectors
        self.arrow_scale = arrow_scale
        self.arrow_max_len = arrow_max_len
        self.arrow_min_mag = arrow_min_mag
        self.arrow_cylinder_radius = arrow_cylinder_radius
        self.arrow_cone_radius = arrow_cone_radius

        self._patch_colors = _taxel_patch_colors()
        self._force_vmax = 1.0
        self._deform = True
        self._vectors = show_force_vectors
        self._arrows_added = False

        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(
            window_name="Xela taxel FK + forces — F deform, V vectors, Q quit",
            width=1280,
            height=900,
        )
        self._spheres = o3d.geometry.TriangleMesh()
        self._arrows = o3d.geometry.TriangleMesh()
        self._base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        self._vis.add_geometry(self._base_frame)
        self._view_initialized = False
        self._closed = False

        self._vis.register_key_callback(ord("F"), self._on_toggle_deform)
        self._vis.register_key_callback(ord("V"), self._on_toggle_vectors)
        self._vis.register_key_callback(ord("Q"), self._on_quit)

        opt = self._vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.08, 0.08, 0.1])

        _get_kinematic_chain(None)

    def _on_toggle_deform(self, vis):
        self._deform = not self._deform
        return False

    def _on_toggle_vectors(self, vis):
        self._vectors = not self._vectors
        return False

    def _on_quit(self, vis):
        self.close()
        return False

    def _frame_geometry(self, result: FkTaxelResult):
        pos = result.positions_deformed if self._deform and result.positions_deformed is not None else result.positions
        forces_world = result.forces_world
        if result.forces_local is not None:
            mag = np.linalg.norm(result.forces_local, axis=-1)
            self._force_vmax = max(float(np.percentile(mag, 98)), 1e-6)
            if self.color_by_force:
                cols = _force_magnitude_colors(result.forces_local, vmax=self._force_vmax)
            else:
                cols = self._patch_colors
        else:
            cols = self._patch_colors
        return pos, cols, forces_world

    def update(self, result: FkTaxelResult) -> None:
        pos, cols, forces_world = self._frame_geometry(result)
        new_spheres = _make_taxel_spheres(pos, cols, radius=self.sphere_radius)
        vectors = self._vectors and forces_world is not None
        new_arrows = (
            _make_force_arrows(
                pos,
                forces_world,
                arrow_scale=self.arrow_scale,
                max_len=self.arrow_max_len,
                min_mag=self.arrow_min_mag,
                cylinder_radius=self.arrow_cylinder_radius,
                cone_radius=self.arrow_cone_radius,
            )
            if vectors
            else __import__("open3d").geometry.TriangleMesh()
        )

        if len(self._spheres.vertices) == 0:
            self._spheres = new_spheres
            self._vis.add_geometry(self._spheres, reset_bounding_box=not self._view_initialized)
        else:
            self._vis.remove_geometry(self._spheres, reset_bounding_box=False)
            self._spheres = new_spheres
            self._vis.add_geometry(self._spheres, reset_bounding_box=False)

        if self._arrows_added:
            self._vis.remove_geometry(self._arrows, reset_bounding_box=False)
            self._arrows_added = False
        self._arrows = new_arrows
        if vectors and len(new_arrows.vertices) > 0:
            self._vis.add_geometry(self._arrows, reset_bounding_box=False)
            self._arrows_added = True

        if not self._view_initialized:
            center = pos.mean(axis=0)
            ctr = self._vis.get_view_control()
            ctr.set_lookat(center.tolist())
            ctr.set_front([-0.55, -0.75, 0.35])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.55)
            self._view_initialized = True

    def poll(self) -> bool:
        """Process Open3D events. Returns False when the window was closed."""
        if self._closed:
            return False
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        if not alive:
            self._closed = True
        return alive

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._vis.destroy_window()
        except Exception:
            pass


def create_fk_taxel_visualizer(**kwargs) -> FkTaxelVisualizer:
    """Open a live Open3D taxel FK viewer."""
    return FkTaxelVisualizer(**kwargs)


def visualize_fk_taxels(
    result: FkTaxelResult,
    *,
    blocking: bool = True,
    **kwargs,
) -> None:
    """Visualize a single FK frame. Set ``blocking=False`` to reuse a visualizer."""
    if blocking:
        viz = create_fk_taxel_visualizer(**kwargs)
        try:
            viz.update(result)
            while viz.poll():
                pass
        finally:
            viz.close()
    else:
        raise ValueError("Use create_fk_taxel_visualizer() for non-blocking updates")
