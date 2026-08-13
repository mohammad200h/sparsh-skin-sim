"""Leap+XELA Gymnasium env: joint actions → flex / FK taxel observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from env.domain_randomization import DomainRandomizationConfig, TetrisSpawn
from util.fk_taxel_util import (
    LEAP_JOINT_ORDER,
    N_TAXELS,
    compute_fk_taxels,
    flex_forces_to_taxel_forces,
    read_leap_joint_angles,
)
from util.flex_util import (
    AllFlexForceEstimator,
    flex_joint_displacements,
    list_flex_names,
)
from util.objects_util import add_tetris_part

SCENE_XML = (
    Path(__file__).resolve().parent.parent
    / "leapXELA_model"
    / "scene_mjx_cube_CoACD_mjx_flex_sensor.xml"
)


def make_leap_flex_env(**spec: Any) -> LeapFlexEnv:
    """Picklable factory for ``AsyncVectorEnv`` workers."""
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    return LeapFlexEnv(**spec)


SOLVER_ITERATIONS = 50
TH_AXL_ACT_INITIAL = 1.6


class LeapFlexEnv(gym.Env):
    """Minimal MuJoCo env driven by Leap joint targets.

    ``action``
        Shape ``(16,)`` joint position targets in ``LEAP_JOINT_ORDER``
        (same order as the 16 position actuators). Written to ``data.ctrl``.

    ``observation``
        Dict with:

        - ``flex_dist``: ``{flex_name: (n_vert, 3)}`` vertex displacements
        - ``flex_force``: ``{flex_name: (n_vert, 3)}`` Kelvin–Voigt forces
        - ``flex_taxel_fk``: FK taxel pack
          ``positions``, ``rotations``, ``forces_local``, ``forces_world``,
          ``positions_deformed`` (all keyed by FK taxel order, 368 taxels)

    Reward is always ``0``. ``truncated`` is always ``False``. When
    ``joint_movement_threshold`` is set, ``terminated`` becomes ``True`` once
    the L2 norm of the per-step joint-angle change drops below that value
    (radians).

    Domain randomization varies tetris spawn parameters. With ``env_slot`` set
    (vector env), the slot picks deterministically from the DR pools once at
    startup. Without ``env_slot``, a new spawn is sampled from the pools on
    every ``reset()``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_xml: Path | str | None = None,
        *,
        n_substeps: int = 1,
        solver_iterations: int = SOLVER_ITERATIONS,
        th_axl_initial: float = TH_AXL_ACT_INITIAL,
        force_window: int = 5,
        spawn_tetris: bool = False,
        tetris_shape: str = "T",
        tetris_scale: float = 1.5,
        tetris_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tetris_flex: str | None = None,
        joint_movement_threshold: float | None = None,
        domain_randomization: DomainRandomizationConfig | None = None,
        env_slot: int | None = None,
    ) -> None:
        super().__init__()
        self._scene_xml = Path(scene_xml) if scene_xml is not None else SCENE_XML
        if not self._scene_xml.is_file():
            raise FileNotFoundError(f"Scene not found: {self._scene_xml}")

        self._n_substeps = int(n_substeps)
        if self._n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")

        self._solver_iterations = int(solver_iterations)
        self._th_axl_initial = float(th_axl_initial)
        self._force_window = int(force_window)
        self._spawn_tetris = bool(spawn_tetris)
        self._default_spawn = TetrisSpawn(
            shape=str(tetris_shape),
            scale=float(tetris_scale),
            offset=tuple(float(v) for v in tetris_offset),
            flex=tetris_flex,
        )
        self._domain_randomization = domain_randomization
        self._env_slot = env_slot
        self._randomize_on_reset = (
            self._spawn_tetris
            and self._domain_randomization is not None
            and self._env_slot is None
        )

        if joint_movement_threshold is not None and joint_movement_threshold < 0:
            raise ValueError("joint_movement_threshold must be >= 0")
        self._joint_movement_threshold = joint_movement_threshold
        self._prev_joint_angles: np.ndarray | None = None
        self._tetris_body_name: str | None = None
        self._current_spawn = self._default_spawn

        self._compile_scene(self._resolve_spawn(None))

    @property
    def tetris_spawn(self) -> dict[str, Any]:
        return self._current_spawn.as_dict()

    def _resolve_spawn(self, rng: np.random.Generator | None) -> TetrisSpawn:
        if not self._spawn_tetris or self._domain_randomization is None:
            return self._default_spawn
        if self._env_slot is not None:
            return self._domain_randomization.for_slot(self._env_slot)
        if rng is None:
            return self._domain_randomization.sample(
                np.random.default_rng()
            )
        return self._domain_randomization.sample(rng)

    def _compile_scene(self, spawn: TetrisSpawn) -> None:
        spec = mj.MjSpec.from_file(self._scene_xml.as_posix())
        self._tetris_body_name = None
        if self._spawn_tetris:
            piece = add_tetris_part(
                spec,
                shape=spawn.shape,
                above_palm=spawn.flex is None,
                flex_name=spawn.flex,
                scale=spawn.scale,
                offset=spawn.offset,
                euler=(0.0, 0.0, np.pi / 4),
            )
            self._tetris_body_name = piece.name

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
            n_vert = int(self.model.flex_vertnum[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_FLEX, name
            )])
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if self._randomize_on_reset:
            spawn = self._resolve_spawn(self.np_random)
            if spawn != self._current_spawn:
                self._compile_scene(spawn)

        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        if self._th_axl_act_id >= 0:
            self._set_actuator_qpos(self._th_axl_act_id, self._th_axl_initial)
        if options is not None and "qpos" in options:
            self.data.qpos[:] = np.asarray(options["qpos"], dtype=np.float64)
        if options is not None and "ctrl" in options:
            self._set_ctrl(options["ctrl"])
        mj.mj_forward(self.model, self.data)
        self._force_est.reset()
        self._prev_joint_angles = read_leap_joint_angles(self.model, self.data)
        obs = self._get_obs()
        return obs, {"tetris_spawn": self.tetris_spawn}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._set_ctrl(action)
        for _ in range(self._n_substeps):
            mj.mj_step(self.model, self.data)
        obs = self._get_obs()
        done = self._terminate()
        self._prev_joint_angles = read_leap_joint_angles(self.model, self.data)
        return obs, 0.0, done, False, {}

    def close(self) -> None:
        return None

    def _terminate(self) -> bool:
        if self._joint_movement_threshold is None:
            return False
        assert self._prev_joint_angles is not None
        joint_angles = read_leap_joint_angles(self.model, self.data)
        movement = float(np.linalg.norm(joint_angles - self._prev_joint_angles))
        return movement < self._joint_movement_threshold
