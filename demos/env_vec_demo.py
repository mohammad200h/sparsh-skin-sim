"""Demo: vectorized LeapFlexEnv with tetris on palm + random joint actions."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import _bootstrap  # noqa: F401
from env import LeapFlexEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SyncVectorEnv demo for LeapFlexEnv"
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-substeps", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use AsyncVectorEnv instead of SyncVectorEnv",
    )
    return parser.parse_args()


def make_env(seed: int, n_substeps: int, scale: float):
    def _thunk():
        return LeapFlexEnv(
            n_substeps=n_substeps,
            spawn_tetris=True,
            tetris_shape="T",
            tetris_scale=scale,
            tetris_flex=None,
        )

    return _thunk


def _peak_force_batch(obs: dict) -> np.ndarray:
    """Peak |f| per env, shape (num_envs,)."""
    peaks = None
    for force in obs["flex_force"].values():
        # force: (num_envs, n_vert, 3)
        mag = np.linalg.norm(force, axis=-1).max(axis=-1)
        peaks = mag if peaks is None else np.maximum(peaks, mag)
    return peaks


def main() -> None:
    args = parse_args()
    factories = [
        make_env(args.seed + i, args.n_substeps, args.scale)
        for i in range(args.num_envs)
    ]
    vec_cls = gym.vector.AsyncVectorEnv if args.use_async else gym.vector.SyncVectorEnv
    vec = vec_cls(factories)

    obs, _ = vec.reset(seed=args.seed)
    print(f"Vector env: {vec_cls.__name__}  num_envs={args.num_envs}")
    print(f"  action batch shape: {vec.action_space.sample().shape}")
    print(f"  fk positions shape: {obs['flex_taxel_fk']['positions'].shape}")
    print(f"  peak |flex_force| at reset: {_peak_force_batch(obs)}")

    try:
        for t in range(args.steps):
            actions = vec.action_space.sample()
            obs, rewards, terminated, truncated, infos = vec.step(actions)
            if t % 5 == 0 or t == args.steps - 1:
                print(
                    f"  t={t:3d}  peak_force={np.round(_peak_force_batch(obs), 5)}  "
                    f"reward={rewards}"
                )
    finally:
        vec.close()
    print("done")


if __name__ == "__main__":
    main()
