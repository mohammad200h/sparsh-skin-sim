"""Collect Leap+XELA cyringe episodes driven by a YAML policy."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.project_venv import exec_project_venv

exec_project_venv()

import argparse
import json
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import mujoco.viewer
import numpy as np
import yaml
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from data_collection.policies import (
    CyringeMotionParams,
    CyringePolicy,
    RandomMotionParams,
    RandomPolicy,
    VectorCyringePolicy,
    VectorRandomPolicy,
)
from env.domain_randomization.cytinge_dr import CyringeDomainRandomizationConfig
from env.leap_flex_cyringe_env import LeapFlexCyringeEnv
from util.fk_taxel_util import LEAP_JOINT_ORDER, N_TAXELS
from util.trajectory_generation import DEFAULT_TRAJ_WAYPOINTS

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "cyringe.yaml"
POLICY_CYRINGE = "cyringe_two_phase"
POLICY_RANDOM = "random"
_CYRINGE_POLICY_ALIASES = frozenset({POLICY_CYRINGE, "cyringe"})
_SINGLE_POLICIES = (CyringePolicy, RandomPolicy)
_VECTOR_POLICIES = (VectorCyringePolicy, VectorRandomPolicy)
_ENV_TERMINATION_CAUSES = frozenset(
    {
        "flex_exploded",
        "sim_unstable",
        "max_timestep",
        "cyringe_dropped",
        "cyringe_flying",
        "cyringe_handle_fully_pressed",
    }
)
_DISCARD_SECTION_KEYS = ("discarded_episode", "discared_episode")


def _env_step_seconds(env: LeapFlexCyringeEnv) -> float:
    return float(env.model.opt.timestep * env._n_substeps)


def _require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config must contain a '{name}' mapping")
    return section


def _discard_section(
    config: dict[str, Any], collection: dict[str, Any], env_cfg: dict[str, Any]
) -> dict[str, Any] | None:
    for mapping in (env_cfg, collection, config):
        for key in _DISCARD_SECTION_KEYS:
            raw = mapping.get(key)
            if raw is not None:
                return raw
    return None


def _parse_discard_termination_causes(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, dict):
        raise ValueError("discarded_episode must be a mapping")
    causes = raw.get("termination_causes", [])
    if causes is None:
        return frozenset()
    if isinstance(causes, str):
        causes = [causes]
    if not isinstance(causes, (list, tuple)):
        raise ValueError(
            "discarded_episode.termination_causes must be a list of strings"
        )
    parsed: list[str] = []
    unknown: list[str] = []
    for cause in causes:
        name = str(cause).strip()
        if not name:
            continue
        if name not in _ENV_TERMINATION_CAUSES:
            unknown.append(name)
            continue
        parsed.append(name)
    if unknown:
        raise ValueError(
            "Unknown discarded_episode.termination_causes "
            f"{unknown}. Expected a subset of {sorted(_ENV_TERMINATION_CAUSES)}"
        )
    return frozenset(parsed)


def _causes_from_info(
    info: dict[str, Any] | None, env_id: int | None = None
) -> list[str]:
    if not info:
        return []
    raw = info.get("termination_causes")
    if env_id is not None and raw is not None:
        raw = raw[env_id]
    if raw is None:
        cause = info.get("termination_cause")
        if env_id is not None and cause is not None:
            cause = cause[env_id]
        if cause is None or cause is False:
            return []
        text = str(cause)
        if not text or text == "None":
            return []
        return [part for part in text.split(",") if part]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(cause) for cause in raw]


def _should_discard_episode(
    causes: list[str], discard_causes: frozenset[str]
) -> bool:
    return bool(discard_causes) and any(cause in discard_causes for cause in causes)


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    collection = _require_section(config, "collection")
    env_cfg = _require_section(config, "env")
    policy_cfg = _require_section(config, "policy")

    episodes = int(collection["episodes"])
    if episodes < 1:
        raise ValueError("collection.episodes must be at least 1")

    duration = float(collection["duration"])
    if duration <= 0.0:
        raise ValueError("collection.duration must be positive")

    n_substeps = int(env_cfg.get("n_substeps", 1))
    if n_substeps < 1:
        raise ValueError("env.n_substeps must be at least 1")

    num_envs = int(env_cfg.get("num_envs", 1))
    if num_envs < 1:
        raise ValueError("env.num_envs must be at least 1")

    max_timestep = env_cfg.get("max_timestep")
    if max_timestep is not None:
        max_timestep = int(max_timestep)
        if max_timestep < 1:
            raise ValueError("env.max_timestep must be at least 1")

    motion_params: CyringeMotionParams | RandomMotionParams
    policy_type = str(policy_cfg.get("type", POLICY_CYRINGE)).strip()
    if policy_type in _CYRINGE_POLICY_ALIASES:
        policy_type = POLICY_CYRINGE
        motion_params = CyringeMotionParams.from_dict(policy_cfg)
        use_trajectory = bool(motion_params.use_trajectory)
        n_traj_waypoints = (
            int(motion_params.traj_waypoints) if use_trajectory else None
        )
        if use_trajectory and (n_traj_waypoints is None or n_traj_waypoints < 1):
            n_traj_waypoints = DEFAULT_TRAJ_WAYPOINTS
        ik_helpers = True
    elif policy_type == POLICY_RANDOM:
        motion_params = RandomMotionParams.from_dict(policy_cfg)
        n_traj_waypoints = None
        ik_helpers = False
    else:
        raise ValueError(
            f"policy.type must be '{POLICY_CYRINGE}' or '{POLICY_RANDOM}', "
            f"got {policy_type!r}"
        )

    domain_randomization = CyringeDomainRandomizationConfig.from_dict(
        env_cfg.get("domain_randomization")
    )
    cyringe_overrides = dict(env_cfg.get("cyringe") or {})
    if (
        domain_randomization is not None
        and domain_randomization.randomize_cyringe
        and bool(cyringe_overrides.get("sticky", False))
    ):
        raise ValueError(
            "Cyringe pose domain randomization requires env.cyringe.sticky=false"
        )

    base_env = {
        "n_substeps": n_substeps,
        "max_timestep": max_timestep,
        "cyringe": cyringe_overrides or None,
        "ik_helpers": ik_helpers,
        "n_traj_waypoints": n_traj_waypoints,
        "domain_randomization": domain_randomization,
    }

    return {
        "config_path": config_path.resolve(),
        "episodes": episodes,
        "duration": duration,
        "output_dir": Path(collection["output_dir"]),
        "seed": collection.get("seed"),
        "render": bool(env_cfg.get("render", False)),
        "num_envs": num_envs,
        "env_kwargs": base_env,
        "policy_type": policy_type,
        "motion_params": motion_params,
        "discard_termination_causes": _parse_discard_termination_causes(
            _discard_section(config, collection, env_cfg)
        ),
    }


def build_collection_env(
    cfg: dict[str, Any],
) -> tuple[LeapFlexCyringeEnv | SyncVectorEnv, int]:
    num_envs = cfg["num_envs"]
    env_kwargs: dict[str, Any] = cfg["env_kwargs"]

    if num_envs == 1:
        return LeapFlexCyringeEnv(**env_kwargs), 1

    factories = [lambda kw=env_kwargs: LeapFlexCyringeEnv(**kw) for _ in range(num_envs)]
    # DISABLED: CyringePolicy must rebuild motion after a real env.reset().
    vec = SyncVectorEnv(
        factories,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    return vec, num_envs


def _reference_env(env: LeapFlexCyringeEnv | SyncVectorEnv) -> LeapFlexCyringeEnv:
    if isinstance(env, SyncVectorEnv):
        sub_env = env.envs[0]
        if not isinstance(sub_env, LeapFlexCyringeEnv):
            raise TypeError("Expected LeapFlexCyringeEnv sub-environments")
        return sub_env
    return env


def _domain_randomization_info(env: LeapFlexCyringeEnv) -> dict[str, Any] | None:
    sample = env._episode_dr
    return None if sample is None else sample.as_dict()


def _cyringe_policy_for_env(
    env: LeapFlexCyringeEnv, motion_params: CyringeMotionParams
) -> CyringePolicy:
    return CyringePolicy(
        env.model,
        env.data,
        env._actuator_ids,
        ee_site=env.ee_site,
        goal_mocap=env.goal_mocap,
        traj_mocap_names=env.traj_mocap_names,
        params=motion_params,
    )


def _random_policy_for_env(
    env: LeapFlexCyringeEnv,
    motion_params: RandomMotionParams,
    seed: int | None,
) -> RandomPolicy:
    return RandomPolicy(
        env.action_space.low,
        env.action_space.high,
        params=motion_params,
        seed=seed,
    )


def _build_policy(
    env: LeapFlexCyringeEnv | SyncVectorEnv,
    num_envs: int,
    motion_params: CyringeMotionParams | RandomMotionParams,
    *,
    policy_type: str,
    seed: int | None,
) -> CyringePolicy | VectorCyringePolicy | RandomPolicy | VectorRandomPolicy:
    if policy_type == POLICY_RANDOM:
        if not isinstance(motion_params, RandomMotionParams):
            raise TypeError("Expected RandomMotionParams for policy.type=random")
        if num_envs == 1:
            if not isinstance(env, LeapFlexCyringeEnv):
                raise TypeError("Expected LeapFlexCyringeEnv for num_envs=1")
            return _random_policy_for_env(env, motion_params, seed)
        if not isinstance(env, SyncVectorEnv):
            raise TypeError("Expected SyncVectorEnv for num_envs > 1")
        policies = [
            _random_policy_for_env(
                sub,  # type: ignore[arg-type]
                motion_params,
                None if seed is None else int(seed) + i,
            )
            for i, sub in enumerate(env.envs)
        ]
        return VectorRandomPolicy(policies)

    if policy_type != POLICY_CYRINGE:
        raise ValueError(f"Unsupported policy_type {policy_type!r}")
    if not isinstance(motion_params, CyringeMotionParams):
        raise TypeError("Expected CyringeMotionParams for cyringe_two_phase")

    if num_envs == 1:
        if not isinstance(env, LeapFlexCyringeEnv):
            raise TypeError("Expected LeapFlexCyringeEnv for num_envs=1")
        return _cyringe_policy_for_env(env, motion_params)

    if not isinstance(env, SyncVectorEnv):
        raise TypeError("Expected SyncVectorEnv for num_envs > 1")
    policies = [
        _cyringe_policy_for_env(sub, motion_params)  # type: ignore[arg-type]
        for sub in env.envs
    ]
    return VectorCyringePolicy(policies)


def _slice_obs(obs: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "flex_dist": {name: values[index] for name, values in obs["flex_dist"].items()},
        "flex_force": {
            name: values[index] for name, values in obs["flex_force"].items()
        },
        "flex_taxel_fk": {
            key: values[index] for key, values in obs["flex_taxel_fk"].items()
        },
    }


def _stack_flex_dict(steps: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not steps:
        return {}
    names = steps[0].keys()
    return {
        name: np.stack([step[name] for step in steps], axis=0)
        for name in names
    }


def _stack_fk(steps: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = steps[0].keys()
    return {
        key: np.stack([step[key] for step in steps], axis=0)
        for key in keys
    }


def _episode_arrays(
    actions: list[np.ndarray],
    obs_steps: list[dict[str, Any]],
    rewards: list[float],
    *,
    step_seconds: float,
) -> dict[str, np.ndarray]:
    if len(rewards) != len(actions):
        raise ValueError(
            f"reward length {len(rewards)} != action length {len(actions)}"
        )
    times = np.arange(len(actions), dtype=np.float64) * step_seconds
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "actions": np.stack(actions, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }

    flex_dist = _stack_flex_dict([obs["flex_dist"] for obs in obs_steps])
    flex_force = _stack_flex_dict([obs["flex_force"] for obs in obs_steps])
    for name, values in flex_dist.items():
        arrays[f"flex_dist/{name}"] = values
    for name, values in flex_force.items():
        arrays[f"flex_force/{name}"] = values

    fk = _stack_fk([obs["flex_taxel_fk"] for obs in obs_steps])
    for key, values in fk.items():
        arrays[f"flex_taxel_fk/{key}"] = values

    return arrays


def _save_episode(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        joint_names=np.asarray(LEAP_JOINT_ORDER),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


@dataclass
class _EpisodeBuffer:
    actions: list[np.ndarray] = field(default_factory=list)
    obs_steps: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def append(
        self, action: np.ndarray, obs: dict[str, Any], reward: float
    ) -> None:
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.obs_steps.append(obs)
        self.rewards.append(float(np.asarray(reward).reshape(())))

    def clear(self) -> None:
        self.actions.clear()
        self.obs_steps.clear()
        self.rewards.clear()

    def __len__(self) -> int:
        return len(self.actions)


def _collect_single_env(
    policy: CyringePolicy | RandomPolicy,
    env: LeapFlexCyringeEnv,
    num_episodes: int,
    *,
    max_steps_per_episode: int,
    output_dir: Path,
    seed: int | None,
    duration: float,
    config_path: Path | None,
    render: bool,
    metadata_base: dict[str, Any],
    discard_termination_causes: frozenset[str],
) -> Path:
    del duration, config_path
    step_seconds = metadata_base["step_seconds"]
    viewer_ctx = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if render
        else nullcontext()
    )

    episodes_saved = 0
    attempts = 0
    discarded = 0
    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    with viewer_ctx as viewer:
        while episodes_saved < num_episodes:
            if render and not viewer.is_running():
                print("Viewer closed; stopping collection early.")
                break

            episode_seed = None if seed is None else int(seed) + attempts
            attempts += 1
            obs, reset_info = env.reset(seed=episode_seed)
            policy.reset()

            buffer = _EpisodeBuffer()
            terminated_by_env = False
            step_info: dict[str, Any] = {}

            for _ in range(max_steps_per_episode):
                if render and not viewer.is_running():
                    break

                step_start = time.time()
                action = policy.act(obs)
                obs, reward, terminated, truncated, step_info = env.step(action)
                buffer.append(action, obs, reward)

                if render:
                    viewer.sync()
                    leftover = step_seconds - (time.time() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)

                if terminated or truncated:
                    terminated_by_env = True
                    break

            if render and not viewer.is_running():
                print("Viewer closed; stopping collection early.")
                break

            if not buffer:
                break

            causes = _causes_from_info(step_info)
            if _should_discard_episode(causes, discard_termination_causes):
                discarded += 1
                pbar.set_postfix(discarded=discarded)
                continue

            metadata = {
                **metadata_base,
                "episode": episodes_saved,
                "seed": episode_seed,
                "num_steps": len(buffer),
                "terminated_by_env": terminated_by_env,
                "termination_causes": causes,
                "env_slot": 0,
                "cyringe_spawn": reset_info.get("cyringe_spawn", env.cyringe_spawn),
                "domain_randomization": reset_info.get("domain_randomization"),
            }
            arrays = _episode_arrays(
                buffer.actions,
                buffer.obs_steps,
                buffer.rewards,
                step_seconds=step_seconds,
            )
            _save_episode(
                output_dir / f"episode_{episodes_saved:04d}.npz",
                arrays,
                metadata=metadata,
            )
            episodes_saved += 1
            pbar.update(1)
            if discarded:
                pbar.set_postfix(discarded=discarded)

    pbar.close()
    return output_dir


def _collect_vector_env(
    policy: VectorCyringePolicy | VectorRandomPolicy,
    env: SyncVectorEnv,
    num_envs: int,
    num_episodes: int,
    *,
    max_steps_per_episode: int,
    output_dir: Path,
    seed: int | None,
    config_path: Path | None,
    metadata_base: dict[str, Any],
    discard_termination_causes: frozenset[str],
) -> Path:
    del config_path

    step_seconds = metadata_base["step_seconds"]
    buffers = [_EpisodeBuffer() for _ in range(num_envs)]
    episodes_saved = 0
    discarded = 0
    attempts = num_envs

    obs, _ = env.reset(seed=seed)
    policy.reset()
    slot_spawns = [env.envs[i].cyringe_spawn for i in range(num_envs)]
    slot_dr = [_domain_randomization_info(env.envs[i]) for i in range(num_envs)]

    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    while episodes_saved < num_episodes:
        actions = policy.act(obs)
        obs, rewards, terminated, truncated, infos = env.step(actions)

        reset_mask = np.zeros(num_envs, dtype=np.bool_)
        reset_seeds: list[int | None] = [None] * num_envs

        for env_id in range(num_envs):
            if episodes_saved >= num_episodes:
                break

            buffers[env_id].append(
                actions[env_id], _slice_obs(obs, env_id), rewards[env_id]
            )
            done = bool(terminated[env_id] or truncated[env_id])
            at_max_steps = len(buffers[env_id]) >= max_steps_per_episode

            if not done and not at_max_steps:
                continue

            causes = _causes_from_info(infos, env_id)
            reset_mask[env_id] = True
            reset_seeds[env_id] = None if seed is None else int(seed) + attempts
            attempts += 1

            if _should_discard_episode(causes, discard_termination_causes):
                discarded += 1
                pbar.set_postfix(discarded=discarded)
                buffers[env_id].clear()
                continue

            episode_seed = None if seed is None else int(seed) + episodes_saved
            metadata = {
                **metadata_base,
                "episode": episodes_saved,
                "seed": episode_seed,
                "num_steps": len(buffers[env_id]),
                "terminated_by_env": done,
                "termination_causes": causes,
                "env_slot": env_id,
                "cyringe_spawn": slot_spawns[env_id],
                "domain_randomization": slot_dr[env_id],
            }
            arrays = _episode_arrays(
                buffers[env_id].actions,
                buffers[env_id].obs_steps,
                buffers[env_id].rewards,
                step_seconds=step_seconds,
            )
            _save_episode(
                output_dir / f"episode_{episodes_saved:04d}.npz",
                arrays,
                metadata=metadata,
            )
            buffers[env_id].clear()
            episodes_saved += 1
            pbar.update(1)
            if discarded:
                pbar.set_postfix(discarded=discarded)

        if not reset_mask.any():
            continue

        # Resetting sub-envs directly leaves SyncVectorEnv's done flags set, and
        # AutoresetMode.DISABLED refuses to step a slot that still looks done.
        obs, _ = env.reset(seed=reset_seeds, options={"reset_mask": reset_mask})
        for env_id in np.flatnonzero(reset_mask).tolist():
            slot_spawns[env_id] = env.envs[env_id].cyringe_spawn
            slot_dr[env_id] = _domain_randomization_info(env.envs[env_id])
            policy.reset(env_id)

    pbar.close()
    return output_dir


def collect_cyringe_data(
    policy: (
        CyringePolicy | VectorCyringePolicy | RandomPolicy | VectorRandomPolicy
    ),
    env: LeapFlexCyringeEnv | SyncVectorEnv,
    num_envs: int,
    num_episodes: int,
    *,
    max_steps_per_episode: int | None = None,
    output_dir: Path | str = "data/cyringe",
    seed: int | None = 0,
    duration: float = 10.0,
    config_path: Path | None = None,
    render: bool = False,
    discard_termination_causes: frozenset[str] | None = None,
) -> Path:
    """Run until ``num_episodes`` are saved (discarded rollouts are retried)."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    output_dir = Path(output_dir)
    ref_env = _reference_env(env)
    step_seconds = _env_step_seconds(ref_env)
    if max_steps_per_episode is None:
        max_steps_per_episode = max(1, int(np.ceil(duration / step_seconds)))
    discard_causes = frozenset(discard_termination_causes or ())

    metadata_base = {
        "config_path": str(config_path) if config_path is not None else None,
        "motion": getattr(policy, "motion_name", POLICY_CYRINGE),
        "profile": asdict(policy.profile),
        "step_seconds": step_seconds,
        "max_steps_per_episode": max_steps_per_episode,
        "max_duration_seconds": duration,
        "num_envs": num_envs,
        "n_taxels": N_TAXELS,
        "joint_names": LEAP_JOINT_ORDER,
        "domain_randomization_enabled": ref_env._domain_randomization is not None,
        "discard_termination_causes": sorted(discard_causes),
        "ee_site": ref_env.ee_site,
        "goal_mocap": ref_env.goal_mocap,
        "traj_mocap_names": ref_env.traj_mocap_names,
    }

    if num_envs == 1:
        if not isinstance(policy, _SINGLE_POLICIES):
            raise TypeError(
                "Expected CyringePolicy or RandomPolicy for num_envs=1"
            )
        if not isinstance(env, LeapFlexCyringeEnv):
            raise TypeError("Expected LeapFlexCyringeEnv for num_envs=1")
        return _collect_single_env(
            policy,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            output_dir=output_dir,
            seed=seed,
            duration=duration,
            config_path=config_path,
            render=render,
            metadata_base=metadata_base,
            discard_termination_causes=discard_causes,
        )

    if not isinstance(policy, _VECTOR_POLICIES):
        raise TypeError(
            "Expected VectorCyringePolicy or VectorRandomPolicy for num_envs > 1"
        )
    if not isinstance(env, SyncVectorEnv):
        raise TypeError("Expected SyncVectorEnv for num_envs > 1")
    if render:
        warnings.warn(
            "render is disabled when env.num_envs > 1",
            stacklevel=2,
        )

    return _collect_vector_env(
        policy,
        env,
        num_envs,
        num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        output_dir=output_dir,
        seed=seed,
        config_path=config_path,
        metadata_base=metadata_base,
        discard_termination_causes=discard_causes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Leap+XELA cyringe data from a YAML config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env, num_envs = build_collection_env(cfg)
    policy = _build_policy(
        env,
        num_envs,
        cfg["motion_params"],
        policy_type=cfg["policy_type"],
        seed=cfg["seed"],
    )

    output_dir = collect_cyringe_data(
        policy,
        env,
        num_envs,
        cfg["episodes"],
        output_dir=cfg["output_dir"],
        seed=cfg["seed"],
        duration=cfg["duration"],
        config_path=cfg["config_path"],
        render=cfg["render"],
        discard_termination_causes=cfg["discard_termination_causes"],
    )
    print(
        f"Saved episodes to {output_dir.resolve()} "
        f"(motion={cfg['policy_type']}, num_envs={num_envs}, "
        f"max_duration={cfg['duration']}s, config={cfg['config_path']})"
    )
    env.close()


if __name__ == "__main__":
    main()
