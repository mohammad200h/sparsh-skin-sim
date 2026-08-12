"""Collect Leap+XELA episodes driven by a grasp ``Policy``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import gymnasium as gym
import mujoco.viewer
import numpy as np
import yaml
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from data_collection.policies import Policy, VectorPolicy
from env.domain_randomization import DomainRandomizationConfig
from env.leap_flex_env import LeapFlexEnv
from util.fk_taxel_util import LEAP_JOINT_ORDER, N_TAXELS
from util.motion_util import GRASP_PATTERNS, GraspProfile

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "squeeze.yaml"
_PROFILE_FIELDS = frozenset(
    field.name for field in fields(GraspProfile) if field.name != "pattern"
)


def _env_step_seconds(env: LeapFlexEnv) -> float:
    return float(env.model.opt.timestep * env._n_substeps)


def _require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config must contain a '{name}' mapping")
    return section


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

    grasp_pattern = policy_cfg.get("grasp_pattern")
    if grasp_pattern not in GRASP_PATTERNS:
        raise ValueError(
            f"policy.grasp_pattern must be one of {GRASP_PATTERNS}, got {grasp_pattern!r}"
        )

    episodes = int(collection["episodes"])
    if episodes < 1:
        raise ValueError("collection.episodes must be at least 1")

    duration = float(collection["duration"])
    if duration <= 0.0:
        raise ValueError("collection.duration must be positive")

    n_substeps = int(env_cfg["n_substeps"])
    if n_substeps < 1:
        raise ValueError("env.n_substeps must be at least 1")

    num_envs = int(env_cfg.get("num_envs", 1))
    if num_envs < 1:
        raise ValueError("env.num_envs must be at least 1")

    joint_movement_threshold = env_cfg.get("joint_movement_threshold")
    if joint_movement_threshold is not None:
        joint_movement_threshold = float(joint_movement_threshold)
        if joint_movement_threshold < 0:
            raise ValueError("env.joint_movement_threshold must be >= 0")

    domain_randomization = DomainRandomizationConfig.from_dict(
        env_cfg.get("domain_randomization")
    )

    tetris_offset = env_cfg.get("tetris_offset", [0.0, 0.0, 0.0])
    if len(tetris_offset) != 3:
        raise ValueError("env.tetris_offset must contain exactly three values")

    base_env = {
        "n_substeps": n_substeps,
        "spawn_tetris": bool(env_cfg.get("spawn_tetris", False)),
        "tetris_shape": str(env_cfg.get("tetris_shape", "T")),
        "tetris_scale": float(env_cfg.get("tetris_scale", 1.5)),
        "tetris_offset": tuple(float(v) for v in tetris_offset),
        "tetris_flex": env_cfg.get("tetris_flex"),
        "joint_movement_threshold": joint_movement_threshold,
        "domain_randomization": domain_randomization,
    }
    env_specs = [
        {
            **base_env,
            "env_slot": slot if domain_randomization is not None and num_envs > 1 else None,
        }
        for slot in range(num_envs)
    ]

    profile_overrides = {
        key: policy_cfg[key]
        for key in _PROFILE_FIELDS
        if key in policy_cfg
    }

    return {
        "config_path": config_path.resolve(),
        "episodes": episodes,
        "duration": duration,
        "output_dir": Path(collection["output_dir"]),
        "seed": collection.get("seed"),
        "render": bool(env_cfg.get("render", False)),
        "num_envs": num_envs,
        "env_specs": env_specs,
        "grasp_pattern": grasp_pattern,
        "profile_overrides": profile_overrides,
    }


def build_collection_env(cfg: dict[str, Any]) -> tuple[LeapFlexEnv | SyncVectorEnv, int]:
    num_envs = cfg["num_envs"]
    env_specs: list[dict[str, Any]] = cfg["env_specs"]

    if num_envs == 1:
        return LeapFlexEnv(**env_specs[0]), 1

    factories = [lambda spec=spec: LeapFlexEnv(**spec) for spec in env_specs]
    vec = SyncVectorEnv(
        factories,
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    return vec, num_envs


def _reference_env(env: LeapFlexEnv | SyncVectorEnv) -> LeapFlexEnv:
    if isinstance(env, SyncVectorEnv):
        sub_env = env.envs[0]
        if not isinstance(sub_env, LeapFlexEnv):
            raise TypeError("Expected LeapFlexEnv sub-environments")
        return sub_env
    return env


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


def _set_obs_slice(
    obs: dict[str, Any], index: int, obs_i: dict[str, Any]
) -> dict[str, Any]:
    for name, values in obs_i["flex_dist"].items():
        obs["flex_dist"][name][index] = values
    for name, values in obs_i["flex_force"].items():
        obs["flex_force"][name][index] = values
    for key, values in obs_i["flex_taxel_fk"].items():
        obs["flex_taxel_fk"][key][index] = values
    return obs


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
    *,
    step_seconds: float,
) -> dict[str, np.ndarray]:
    times = np.arange(len(actions), dtype=np.float64) * step_seconds
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "actions": np.stack(actions, axis=0),
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

    def append(self, action: np.ndarray, obs: dict[str, Any]) -> None:
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.obs_steps.append(obs)

    def clear(self) -> None:
        self.actions.clear()
        self.obs_steps.clear()

    def __len__(self) -> int:
        return len(self.actions)


def _build_policy(
    env: LeapFlexEnv | SyncVectorEnv,
    num_envs: int,
    grasp_pattern: str,
    profile_overrides: dict[str, Any],
) -> Policy | VectorPolicy:
    ref_env = _reference_env(env)
    params = {
        "model": ref_env.model,
        "timestep": _env_step_seconds(ref_env),
        **profile_overrides,
    }
    if num_envs == 1:
        return Policy(grasp_pattern, params)
    return VectorPolicy(grasp_pattern, params, num_envs)


def _collect_single_env(
    policy: Policy,
    env: LeapFlexEnv,
    num_episodes: int,
    *,
    max_steps_per_episode: int,
    output_dir: Path,
    seed: int | None,
    duration: float,
    config_path: Path | None,
    render: bool,
    metadata_base: dict[str, Any],
) -> Path:
    step_seconds = metadata_base["step_seconds"]
    viewer_ctx = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if render
        else nullcontext()
    )

    with viewer_ctx as viewer:
        for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
            if render and not viewer.is_running():
                print("Viewer closed; stopping collection early.")
                break

            episode_seed = None if seed is None else seed + episode
            obs, reset_info = env.reset(seed=episode_seed)
            policy.reset()

            buffer = _EpisodeBuffer()
            terminated_by_env = False

            for _ in range(max_steps_per_episode):
                if render and not viewer.is_running():
                    break

                step_start = time.time()
                action = policy.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                buffer.append(action, obs)

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

            metadata = {
                **metadata_base,
                "episode": episode,
                "seed": episode_seed,
                "num_steps": len(buffer),
                "terminated_by_env": terminated_by_env,
                "env_slot": 0,
                "tetris_spawn": reset_info.get("tetris_spawn", env.tetris_spawn),
            }
            arrays = _episode_arrays(
                buffer.actions, buffer.obs_steps, step_seconds=step_seconds
            )
            _save_episode(
                output_dir / f"episode_{episode:04d}.npz",
                arrays,
                metadata=metadata,
            )

    return output_dir


def _collect_vector_env(
    policy: VectorPolicy,
    env: SyncVectorEnv,
    num_envs: int,
    num_episodes: int,
    *,
    max_steps_per_episode: int,
    output_dir: Path,
    seed: int | None,
    config_path: Path | None,
    metadata_base: dict[str, Any],
) -> Path:
    del config_path

    step_seconds = metadata_base["step_seconds"]
    buffers = [_EpisodeBuffer() for _ in range(num_envs)]
    episodes_collected = 0
    slot_tetris_spawns = [env.envs[i].tetris_spawn for i in range(num_envs)]

    obs, _ = env.reset(seed=seed)
    policy.reset()

    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    while episodes_collected < num_episodes:
        actions = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(actions)

        for env_id in range(num_envs):
            if episodes_collected >= num_episodes:
                break

            buffers[env_id].append(actions[env_id], _slice_obs(obs, env_id))
            done = bool(terminated[env_id] or truncated[env_id])
            at_max_steps = len(buffers[env_id]) >= max_steps_per_episode

            if not done and not at_max_steps:
                continue

            episode_seed = None if seed is None else seed + episodes_collected
            metadata = {
                **metadata_base,
                "episode": episodes_collected,
                "seed": episode_seed,
                "num_steps": len(buffers[env_id]),
                "terminated_by_env": done,
                "env_slot": env_id,
                "tetris_spawn": slot_tetris_spawns[env_id],
            }
            arrays = _episode_arrays(
                buffers[env_id].actions,
                buffers[env_id].obs_steps,
                step_seconds=step_seconds,
            )
            _save_episode(
                output_dir / f"episode_{episodes_collected:04d}.npz",
                arrays,
                metadata=metadata,
            )
            buffers[env_id].clear()
            policy.reset(env_id)
            episodes_collected += 1
            pbar.update(1)

            if at_max_steps and not done:
                reset_seed = None if episode_seed is None else episode_seed + num_envs
                obs_i, reset_info = env.envs[env_id].reset(seed=reset_seed)
                slot_tetris_spawns[env_id] = reset_info.get(
                    "tetris_spawn", env.envs[env_id].tetris_spawn
                )
                obs = _set_obs_slice(obs, env_id, obs_i)

    pbar.close()
    return output_dir


def collect_squeeze_data(
    policy: Policy | VectorPolicy,
    env: LeapFlexEnv | SyncVectorEnv,
    num_envs: int,
    num_episodes: int,
    *,
    max_steps_per_episode: int | None = None,
    output_dir: Path | str = "data/squeeze",
    seed: int | None = 0,
    duration: float = 10.0,
    config_path: Path | None = None,
    render: bool = False,
) -> Path:
    """Run ``num_episodes`` and write one compressed ``.npz`` per episode."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    output_dir = Path(output_dir)
    ref_env = _reference_env(env)
    step_seconds = _env_step_seconds(ref_env)
    if max_steps_per_episode is None:
        max_steps_per_episode = max(1, int(np.ceil(duration / step_seconds)))

    metadata_base = {
        "config_path": str(config_path) if config_path is not None else None,
        "grasp_pattern": policy.profile.pattern,
        "profile": asdict(policy.profile),
        "step_seconds": step_seconds,
        "max_steps_per_episode": max_steps_per_episode,
        "max_duration_seconds": duration,
        "num_envs": num_envs,
        "domain_randomization_enabled": ref_env._domain_randomization is not None,
        "n_taxels": N_TAXELS,
        "joint_names": LEAP_JOINT_ORDER,
    }

    if num_envs == 1:
        if not isinstance(policy, Policy):
            raise TypeError("Expected Policy for num_envs=1")
        if not isinstance(env, LeapFlexEnv):
            raise TypeError("Expected LeapFlexEnv for num_envs=1")
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
        )

    if not isinstance(policy, VectorPolicy):
        raise TypeError("Expected VectorPolicy for num_envs > 1")
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Leap+XELA data from a YAML config"
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
        cfg["grasp_pattern"],
        cfg["profile_overrides"],
    )

    output_dir = collect_squeeze_data(
        policy,
        env,
        num_envs,
        cfg["episodes"],
        output_dir=cfg["output_dir"],
        seed=cfg["seed"],
        duration=cfg["duration"],
        config_path=cfg["config_path"],
        render=cfg["render"],
    )
    print(
        f"Saved episodes to {output_dir.resolve()} "
        f"(pattern={cfg['grasp_pattern']}, num_envs={num_envs}, "
        f"max_duration={cfg['duration']}s, config={cfg['config_path']})"
    )
    env.close()


if __name__ == "__main__":
    main()
