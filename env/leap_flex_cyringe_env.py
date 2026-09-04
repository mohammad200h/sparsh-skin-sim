"""Leap+XELA Gymnasium env with a cyringe spawned like ``demos/demo_cyringe.py``."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from env.domain_randomization.cytinge_dr import (
    CyringeDomainRandomizationConfig,
    CyringeEpisodeSample,
)
from env.rewad import SqueezeCyringeReward
from util.fk_taxel_util import (
    LEAP_JOINT_ORDER,
    N_TAXELS,
    compute_fk_taxels,
    flex_forces_to_taxel_forces,
    read_leap_joint_angles,
)
from util.flex_util import (
    AllFlexForceEstimator,
    add_flex_center_ee_site,
    flex_joint_displacements,
    list_flex_names,
)
from util.ik_common import (
    DEFAULT_IK_GOAL_MOCAP,
    DEFAULT_THUMB_EE_SITE,
    DEFAULT_THUMB_FLEX,
    add_ik_goal_mocap,
)
from util.objects_util import (
    add_cyringe,
    cyringe_housing_freejoint_id,
    cyringe_housing_pose,
    set_cyringe_pose,
)
from util.trajectory_generation import add_trajectory_mocaps

SCENE_XML = (
    Path(__file__).resolve().parent.parent
    / "leapXELA_model"
    / "scene_mjx_cube_CoACD_mjx_flex_sensor.xml"
)
ENV_CONFIG_JSON = Path(__file__).resolve().parent / "env_config.json"
# Flex skins often explode with huge but finite stretch (no NaNs / BADQ warnings).
FLEX_EXPLODE_STRETCH = 5.0  # edge length / rest length
FLEX_EXPLODE_POS_M = 2.0  # max |vertex| from origin [m]
CYRINGE_FLYING_VZ_M_S = 0.9  # |housing COM vz| [m/s] counts as flying


def load_env_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else ENV_CONFIG_JSON
    with config_path.open() as f:
        return json.load(f)


def _xyz(value: Any, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be an [x, y, z] triplet")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class CyringeSpawn:
    """Resolved cyringe spawn parameters matching ``demo_cyringe.py``."""

    flex: str | None
    scale: float
    offset: tuple[float, float, float]
    euler: tuple[float, float, float]
    sticky: bool
    friction: tuple[float, float, float]
    housing_friction: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "flex": self.flex,
            "scale": self.scale,
            "offset": list(self.offset),
            "euler": list(self.euler),
            "sticky": self.sticky,
            "friction": list(self.friction),
            "housing_friction": list(self.housing_friction),
        }

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> CyringeSpawn:
        flex_raw = raw.get("flex")
        flex = None if flex_raw is None else str(flex_raw)
        offset = _xyz(raw.get("offset", [0.0, 0.0, 0.0]), "cyringe.offset")
        spawn_offset = (
            offset[0] + float(raw.get("delta_x", 0.0)),
            offset[1] + float(raw.get("delta_y", 0.0)),
            offset[2] + float(raw.get("delta_z", 0.0)),
        )
        euler = _xyz(
            raw.get("euler", [0.0, float(np.pi / 2), float(-np.pi / 2)]),
            "cyringe.euler",
        )
        spawn_euler = (
            euler[0],
            euler[1],
            euler[2] + float(raw.get("delta_rot_z", 0.0)),
        )
        return cls(
            flex=flex,
            scale=float(raw.get("scale", 1.0)),
            offset=spawn_offset,
            euler=spawn_euler,
            sticky=bool(raw.get("sticky", False)),
            friction=_xyz(
                raw.get("friction", [0.1, 0.005, 0.0001]),
                "cyringe.friction",
            ),
            housing_friction=_xyz(
                raw.get("housing_friction", [0.0, 0.0, 2.5]),
                "cyringe.housing_friction",
            ),
        )


class LeapFlexCyringeEnv(gym.Env):
    """Minimal MuJoCo env driven by Leap joint targets, with a spawned cyringe.

    ``action``
        Shape ``(16,)`` joint position targets in ``LEAP_JOINT_ORDER``
        (same order as the 16 position actuators). Written to ``data.ctrl``.

    ``observation``
        Dict with ``flex_dist``, ``flex_force``, and ``flex_taxel_fk``.

    Cyringe placement matches ``demos/demo_cyringe.py``: above the palm pads
    (or a named flex), default offset ``(0.01, 0.05, -0.12)``, and a 90°
    side-lay euler ``(0, π/2, -π/2)`` plus optional yaw. Defaults live in
    ``env/env_config.json``.

    Domain randomization (when enabled) samples a new hand qpos and cyringe
    freejoint pose on every ``reset()``. Pose DR requires a free housing
    (``sticky=false``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_xml: Path | str | None = None,
        *,
        config_path: Path | str | None = None,
        n_substeps: int | None = None,
        solver_iterations: int | None = None,
        th_axl_initial: float | None = None,
        force_window: int | None = None,
        max_timestep: int | None = None,
        motion_type: str | None = None,
        cyringe: dict[str, Any] | None = None,
        ik_helpers: bool = True,
        n_traj_waypoints: int | None = None,
        randomize_hand: bool | None = None,
        randomize_cyringe: bool | None = None,
        domain_randomization: (
            CyringeDomainRandomizationConfig | dict[str, Any] | None
        ) = None,
    ) -> None:
        super().__init__()
        cfg = load_env_config(config_path)
        cyringe_cfg = dict(cfg.get("cyringe", {}))
        if cyringe is not None:
            cyringe_cfg.update(cyringe)
        json_dr = cfg.get("domain_randomization")
        if domain_randomization is not None:
            self._domain_randomization = CyringeDomainRandomizationConfig.resolve(
                domain_randomization,
                json_dr,
            )
        else:
            self._domain_randomization = (
                CyringeDomainRandomizationConfig.from_env_config(
                    json_dr,
                    randomize_hand=randomize_hand,
                    randomize_cyringe=randomize_cyringe,
                )
            )
        self._episode_dr: CyringeEpisodeSample | None = None

        self._scene_xml = Path(scene_xml) if scene_xml is not None else SCENE_XML
        if not self._scene_xml.is_file():
            raise FileNotFoundError(f"Scene not found: {self._scene_xml}")

        self._n_substeps = int(
            n_substeps if n_substeps is not None else cfg.get("n_substeps", 1)
        )
        if self._n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")

        self._solver_iterations = int(
            solver_iterations
            if solver_iterations is not None
            else cfg.get("solver_iterations", 50)
        )
        self._th_axl_initial = float(
            th_axl_initial
            if th_axl_initial is not None
            else cfg.get("th_axl_initial", 1.6)
        )
        self._force_window = int(
            force_window if force_window is not None else cfg.get("force_window", 5)
        )
        self._max_timestep = int(
            max_timestep if max_timestep is not None else cfg.get("max_timestep", 200)
        )
        if self._max_timestep < 1:
            raise ValueError("max_timestep must be >= 1")
        self._timestep = 0
        self._reward = SqueezeCyringeReward()

        self._cyringe_housing_name: str | None = None
        self._cyringe_housing_body_id = -1
        self._cyringe_freejoint_id = -1
        self._cyringe_spawn_pos = np.zeros(3, dtype=np.float64)
        self._cyringe_spawn_z = 0.0
        self._squeeze_qposadr = -1
        self._squeeze_qpos_hi = 0.0
        self._ik_helpers = bool(ik_helpers)
        self._n_traj_waypoints = (
            None if n_traj_waypoints is None else int(n_traj_waypoints)
        )
        self.ee_site: str | None = None
        self.goal_mocap: str | None = None
        self.traj_mocap_names: list[str] | None = None
        self._current_spawn = CyringeSpawn.from_config(cyringe_cfg)
        self._compile_scene(self._current_spawn)

    @property
    def cyringe_spawn(self) -> dict[str, Any]:
        return self._current_spawn.as_dict()

    def _compile_scene(self, spawn: CyringeSpawn) -> None:
        spec = mj.MjSpec.from_file(self._scene_xml.as_posix())
        housing, spawn_pos = add_cyringe(
            spec,
            above_palm=spawn.flex is None,
            flex_name=spawn.flex,
            scale=spawn.scale,
            offset=spawn.offset,
            euler=spawn.euler,
            sticky=spawn.sticky,
            friction=spawn.friction,
            housing_friction=spawn.housing_friction,
        )
        self._cyringe_housing_name = housing.name
        self._cyringe_spawn_pos = np.asarray(spawn_pos, dtype=np.float64).copy()
        self._cyringe_spawn_z = float(self._cyringe_spawn_pos[2])

        self.ee_site = None
        self.goal_mocap = None
        self.traj_mocap_names = None
        if self._ik_helpers:
            self.ee_site = add_flex_center_ee_site(
                spec,
                DEFAULT_THUMB_FLEX,
                site_name=DEFAULT_THUMB_EE_SITE,
            )
            self.goal_mocap = add_ik_goal_mocap(spec, name=DEFAULT_IK_GOAL_MOCAP)
            if self._n_traj_waypoints is not None:
                self.traj_mocap_names = add_trajectory_mocaps(
                    spec, n_waypoints=self._n_traj_waypoints
                )

        self.model = spec.compile()
        self.model.opt.iterations = self._solver_iterations
        self.model.opt.tolerance = 0.0
        self.data = mj.MjData(self.model)

        if self.model.nu != len(LEAP_JOINT_ORDER):
            raise ValueError(
                f"Expected {len(LEAP_JOINT_ORDER)} actuators, got {self.model.nu}"
            )
        self._actuator_ids = self._resolve_actuator_ids()
        self._ctrl_lo = self.model.actuator_ctrlrange[self._actuator_ids, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[self._actuator_ids, 1].copy()
        self._th_axl_act_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_ACTUATOR, "th_axl_act"
        )
        self._reward.bind(self.model, self._cyringe_housing_name)
        self._cyringe_freejoint_id = cyringe_housing_freejoint_id(
            self.model, self._cyringe_housing_name
        )
        if (
            self._domain_randomization is not None
            and self._domain_randomization.randomize_cyringe
            and self._cyringe_freejoint_id < 0
        ):
            raise ValueError(
                "Cyringe pose domain randomization requires a free housing; "
                "set sticky=false"
            )
        self._cache_termination_ids()

        self._flex_names = list_flex_names(self.model)
        self._force_est = AllFlexForceEstimator(
            self.model, window=self._force_window, use_qvel=True
        )

        self.action_space = spaces.Box(
            low=self._ctrl_lo.astype(np.float32),
            high=self._ctrl_hi.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()
        self._current_spawn = spawn

    def _resolve_actuator_ids(self) -> np.ndarray:
        ids = np.empty(len(LEAP_JOINT_ORDER), dtype=np.int32)
        for i, joint_name in enumerate(LEAP_JOINT_ORDER):
            act_name = f"{joint_name}_act"
            act_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id < 0:
                raise ValueError(f"Actuator '{act_name}' not found")
            ids[i] = act_id
        return ids

    def _build_observation_space(self) -> spaces.Dict:
        flex_dist: dict[str, spaces.Box] = {}
        flex_force: dict[str, spaces.Box] = {}
        for name in self._flex_names:
            n_vert = int(
                self.model.flex_vertnum[
                    mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_FLEX, name)
                ]
            )
            shape = (n_vert, 3)
            flex_dist[name] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float64
            )
            flex_force[name] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float64
            )

        fk_space = spaces.Dict(
            {
                "positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_TAXELS, 3), dtype=np.float64
                ),
                "rotations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(N_TAXELS, 3, 3),
                    dtype=np.float64,
                ),
                "forces_local": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_TAXELS, 3), dtype=np.float64
                ),
                "forces_world": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_TAXELS, 3), dtype=np.float64
                ),
                "positions_deformed": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(N_TAXELS, 3), dtype=np.float64
                ),
            }
        )
        return spaces.Dict(
            {
                "flex_dist": spaces.Dict(flex_dist),
                "flex_force": spaces.Dict(flex_force),
                "flex_taxel_fk": fk_space,
            }
        )

    def _set_ctrl(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(len(LEAP_JOINT_ORDER))
        action = np.clip(action, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[self._actuator_ids] = action

    def _set_actuator_qpos(self, act_id: int, value: float) -> None:
        joint_id = int(self.model.actuator_trnid[act_id, 0])
        self.data.ctrl[act_id] = float(value)
        self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = float(value)

    def _apply_hand_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(len(LEAP_JOINT_ORDER))
        qpos = np.clip(qpos, self._ctrl_lo, self._ctrl_hi)
        for act_id, value in zip(self._actuator_ids, qpos):
            self._set_actuator_qpos(int(act_id), float(value))

    def _apply_episode_randomization(self, sample: CyringeEpisodeSample) -> None:
        if sample.hand_qpos is not None:
            self._apply_hand_qpos(np.asarray(sample.hand_qpos, dtype=np.float64))
        if self._domain_randomization is None or not (
            self._domain_randomization.randomize_cyringe
        ):
            return
        if self._cyringe_housing_name is None:
            return
        pos = self._cyringe_spawn_pos + np.asarray(
            sample.cyringe_offset, dtype=np.float64
        )
        euler = np.asarray(self._current_spawn.euler, dtype=np.float64) + np.asarray(
            sample.cyringe_euler_delta, dtype=np.float64
        )
        set_cyringe_pose(
            self.model,
            self.data,
            self._cyringe_housing_name,
            pos,
            euler=euler,
        )
        self._cyringe_spawn_z = float(pos[2])

    def _get_obs(self) -> dict[str, Any]:
        forces = self._force_est.update(self.model, self.data)
        flex_dist = {
            name: flex_joint_displacements(self.model, self.data, name)
            for name in self._flex_names
        }
        flex_force = {name: forces[name].copy() for name in self._flex_names}

        joint_angles = read_leap_joint_angles(self.model, self.data)
        taxel_forces = flex_forces_to_taxel_forces(self.model, forces)
        fk = compute_fk_taxels(joint_angles, taxel_forces)

        assert fk.forces_local is not None
        assert fk.forces_world is not None
        assert fk.positions_deformed is not None
        flex_taxel_fk = {
            "positions": np.asarray(fk.positions, dtype=np.float64),
            "rotations": np.asarray(fk.rotations, dtype=np.float64),
            "forces_local": np.asarray(fk.forces_local, dtype=np.float64),
            "forces_world": np.asarray(fk.forces_world, dtype=np.float64),
            "positions_deformed": np.asarray(fk.positions_deformed, dtype=np.float64),
        }
        return {
            "flex_dist": flex_dist,
            "flex_force": flex_force,
            "flex_taxel_fk": flex_taxel_fk,
        }

    def _cache_termination_ids(self) -> None:
        housing = self._cyringe_housing_name or ""
        self._cyringe_housing_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, housing
        )
        prefix = housing.rsplit("/", 1)[0] + "/" if "/" in housing else ""
        squeeze_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, f"{prefix}squeeze"
        )
        if squeeze_id < 0:
            self._squeeze_qposadr = -1
            self._squeeze_qpos_hi = 0.0
            return
        self._squeeze_qposadr = int(self.model.jnt_qposadr[squeeze_id])
        self._squeeze_qpos_hi = float(self.model.jnt_range[squeeze_id, 1])

    def _terminate_max_timestep(self) -> bool:
        return self._timestep >= self._max_timestep

    def _terminate_flex_exploded(self) -> bool:
        """True when a flex skin stretches or flies apart while still finite."""
        if int(self.model.nflexvert) > 0:
            verts = np.asarray(self.data.flexvert_xpos)
            if np.isfinite(verts).all() and float(np.max(np.abs(verts))) > FLEX_EXPLODE_POS_M:
                return True

        if int(self.model.nflexedge) > 0:
            length = np.asarray(self.data.flexedge_length)
            rest = np.asarray(self.model.flexedge_length0)
            if np.isfinite(length).all():
                valid = rest > 1e-9
                if valid.any():
                    stretch = float(np.max(length[valid] / rest[valid]))
                    if stretch > FLEX_EXPLODE_STRETCH:
                        return True
        return False

    def _terminate_sim_unstable(self) -> bool:
        """True when the solver produces NaNs/Infs or BADQ warnings."""
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.qacc).all()
        ):
            return True
        if int(self.model.nflexvert) > 0 and not np.isfinite(
            self.data.flexvert_xpos
        ).all():
            return True
        if int(self.model.nflexedge) > 0 and not np.isfinite(
            self.data.flexedge_length
        ).all():
            return True
        warning = self.data.warning
        return (
            int(warning[mj.mjtWarning.mjWARN_BADQPOS].number)
            + int(warning[mj.mjtWarning.mjWARN_BADQVEL].number)
            + int(warning[mj.mjtWarning.mjWARN_BADQACC].number)
        ) > 0

    def _terminate_cyringe_dropped(self) -> bool:
        if self._cyringe_housing_body_id < 0:
            return False
        z = float(self.data.xpos[self._cyringe_housing_body_id, 2])
        return z < self._cyringe_spawn_z - 0.1

    def _terminate_cyringe_flying(self) -> bool:
        if self._cyringe_housing_body_id < 0:
            return False
        # cvel is (rot, lin) at the body COM; index 5 is world vz.
        vz = float(self.data.cvel[self._cyringe_housing_body_id, 5])
        return abs(vz) > CYRINGE_FLYING_VZ_M_S

    def _terminate_handle_max(self) -> bool:
        if self._squeeze_qposadr < 0:
            return False
        q = float(self.data.qpos[self._squeeze_qposadr])
        return q >= self._squeeze_qpos_hi

    def _termination_causes(self) -> list[str]:
        causes: list[str] = []
        if self._terminate_flex_exploded():
            causes.append("flex_exploded")
        if self._terminate_sim_unstable():
            causes.append("sim_unstable")
        if self._terminate_max_timestep():
            causes.append("max_timestep")
        if self._terminate_cyringe_dropped():
            causes.append("cyringe_dropped")
        if self._terminate_cyringe_flying():
            causes.append("cyringe_flying")
        if self._terminate_handle_max():
            causes.append("handle_max")
        return causes

    def _terminate(self) -> bool:
        return bool(self._termination_causes())

    def _step_info(self, causes: list[str]) -> dict[str, Any]:
        if not causes:
            cause: str | None = None
        elif len(causes) == 1:
            cause = causes[0]
        else:
            cause = ",".join(causes)
        return {
            "termination_cause": cause,
            "termination_causes": causes,
            "flex_exploded": "flex_exploded" in causes,
            "sim_unstable": "sim_unstable" in causes,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        if self._th_axl_act_id >= 0:
            self._set_actuator_qpos(self._th_axl_act_id, self._th_axl_initial)
        self._cyringe_spawn_z = float(self._cyringe_spawn_pos[2])
        self._episode_dr = None
        if self._domain_randomization is not None:
            self._episode_dr = self._domain_randomization.sample(
                self.np_random,
                hand_lo=self._ctrl_lo,
                hand_hi=self._ctrl_hi,
            )
            self._apply_episode_randomization(self._episode_dr)
        self._reward.reset(self.model, self.data)
        if options is not None and "qpos" in options:
            self.data.qpos[:] = np.asarray(options["qpos"], dtype=np.float64)
        if options is not None and "ctrl" in options:
            self._set_ctrl(options["ctrl"])
        mj.mj_forward(self.model, self.data)
        self._force_est.reset()
        self._timestep = 0
        obs = self._get_obs()
        info: dict[str, Any] = {"cyringe_spawn": self.cyringe_spawn}
        if self._episode_dr is not None:
            dr_info = self._episode_dr.as_dict()
            if self._cyringe_housing_name is not None:
                pos, quat = cyringe_housing_pose(
                    self.model, self.data, self._cyringe_housing_name
                )
                dr_info["cyringe_pos"] = pos.tolist()
                dr_info["cyringe_quat"] = quat.tolist()
            info["domain_randomization"] = dr_info
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._set_ctrl(action)
        for _ in range(self._n_substeps):
            mj.mj_step(self.model, self.data)
        self._timestep += 1
        reward = self._reward(self.model, self.data)
        sys.stdout.write(f"  step={self._timestep}/{self._max_timestep}")
        sys.stdout.flush()
        obs = self._get_obs()
        causes = self._termination_causes()
        done = bool(causes)
        return obs, reward, done, False, self._step_info(causes)

    def close(self) -> None:
        return None
