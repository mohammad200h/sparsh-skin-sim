# Cyringe data collection

`data_collection_cyringe.py` rolls out `LeapFlexCyringeEnv` with a YAML-selected policy and writes one compressed `.npz` per saved episode.

Run from the repo root. The script inserts the repo onto `sys.path` and switches into the project venv.

## Run

Default config is `data_collection/config/cyringe.yaml` (two-phase MF/RF close + thumb IK):

```bash
python data_collection/data_collection_cyringe.py
```

Random joint-target policy:

```bash
python data_collection/data_collection_cyringe.py \
  --config data_collection/config/cyringe_random_policy.yaml
```

Any other YAML with the same `collection` / `env` / `policy` sections works:

```bash
python data_collection/data_collection_cyringe.py --config path/to/config.yaml
```

Collection continues until `collection.episodes` files are written (discarded rollouts are retried, not counted). Closing the MuJoCo viewer stops a single-env run early.

## Related files

| Path | Role |
| --- | --- |
| `data_collection/data_collection_cyringe.py` | CLI, YAML load, rollout loop, `.npz` writer |
| `data_collection/config/cyringe.yaml` | Two-phase IK policy (default) |
| `data_collection/config/cyringe_random_policy.yaml` | Uniform random joint targets |
| `data_collection/policies.py` | `CyringePolicy` / `RandomPolicy` and vector wrappers |
| `env/leap_flex_cyringe_env.py` | Gymnasium env (actions, obs, terminations) |
| `env/env_config.json` | Env defaults merged under YAML `env` |

## YAML layout

Three top-level mappings are required: `collection`, `env`, `policy`.

### `collection`

| Key | Meaning |
| --- | --- |
| `episodes` | Number of `.npz` files to save (`>= 1`) |
| `duration` | Max episode wall time in seconds; converted to `ceil(duration / step_seconds)` collector steps |
| `output_dir` | Directory for `episode_0000.npz`, `episode_0001.npz`, … |
| `seed` | Optional int. Episode `i` uses `seed + i` (single env) or a per-slot increment (vector env) |

`step_seconds` is `model.opt.timestep * env.n_substeps`. The env can still end earlier via `env.max_timestep` or another termination cause.

### `env`

| Key | Meaning |
| --- | --- |
| `n_substeps` | Physics steps per env `step()` |
| `max_timestep` | Env horizon; omitted values fall back to `env/env_config.json` |
| `render` | Single-env only: attach a passive MuJoCo viewer and sleep to real time. Ignored (with a warning) when `num_envs > 1` |
| `num_envs` | `1` → `LeapFlexCyringeEnv`; `> 1` → `SyncVectorEnv` with `AutoresetMode.DISABLED` |
| `cyringe` | Overrides merged onto JSON `cyringe` (e.g. `sticky: false`) |
| `domain_randomization` | Passed through to the env. Pose DR needs a free housing (`sticky: false`) |
| `discarded_episode.termination_causes` | List of cause names; matching episodes are not saved |

A misspelling `discared_episode` is still accepted.

YAML `env` values override `env/env_config.json` for the keys the collector forwards (`n_substeps`, `max_timestep`, `cyringe`, `domain_randomization`). Solver iterations, thumb axial init, and spawn pose still come from the JSON unless you put them under `cyringe`.

### `policy`

`policy.type` selects the controller (`cyringe` if omitted):

| `type` | Class | Motion |
| --- | --- | --- |
| `cyringe` or `cyringe_two_phase` | `CyringePolicy` | MF/RF close, then thumb IK toward `cyringe/handle` (same as `demos/demo_cyringe.py`) |
| `random` | `RandomPolicy` | Uniform samples in actuator `ctrlrange`, held for `action_hold` steps |

Unknown types raise. Cyringe policy enables IK helpers (ee site, goal mocap, optional trajectory mocaps). Random policy does not.

#### Two-phase (`cyringe`) keys

| Key | Default / notes |
| --- | --- |
| `ik_backend` | `pyroki` (`dls` / `mink` / `pyroki`) |
| `close_duration` | Seconds to close MF/RF |
| `mcp_target`, `pip_target`, `dip_target` | MF/RF close targets |
| `handle_site` | Default `cyringe/handle` |
| `reach_tol` | IK reach tolerance |
| `delta_ik_z` | Extra handle-frame Z offset for the IK goal |
| `use_trajectory` | If true, thumb follows a push trajectory along handle +Z |
| `traj_stroke` | Stroke length (m); `null` uses the trajectory helper default |
| `traj_waypoints` | Waypoint count (collector default **15** if trajectory is on and the value is missing) |

#### Random keys

| Key | Meaning |
| --- | --- |
| `action_hold` | Env steps to reuse each sampled 16-DoF target (`>= 1`, default **40**) |

## Policies

Both policies are open-loop: `act(obs)` ignores the observation.

**CyringePolicy** rebuilds `cyringe_two_phase_motion` on `reset()` from the current `MjData` (needed after domain randomization). It writes `data.ctrl` while iterating the IK motion and returns the 16 Leap actuator targets.

**RandomPolicy** draws `uniform(low, high)` in the env action space. With a seed, each `reset()` reseeds as `seed + reset_index`.

Vector runs wrap one policy instance per sub-env (`VectorCyringePolicy` / `VectorRandomPolicy`). Random sub-policies get `seed + slot`.

## Episode loop

An episode ends on the first of:

1. Env `terminated` / `truncated` (any cause in `_termination_causes`)
2. Collector step count `>= ceil(duration / step_seconds)`

Vector slots that finish are reset with `options={"reset_mask": ...}` so `AutoresetMode.DISABLED` can keep stepping. Only the finished slot is reseeded.

### Termination causes

From `LeapFlexCyringeEnv._termination_causes()`:

| Cause | When |
| --- | --- |
| `flex_exploded` | Flex edge stretch or vertex fly-away |
| `sim_unstable` | MuJoCo BADQPOS / BADQVEL / BADQACC |
| `max_timestep` | `env.max_timestep` reached |
| `cyringe_dropped` | Housing COM z more than 0.1 m below spawn z |
| `cyringe_flying` | Housing COM `\|vz\|` above `0.9` m/s |
| `handle_max` | Squeeze joint at its upper limit |

If any listed cause is in `env.discarded_episode.termination_causes`, the rollout is dropped and another attempt is run. Progress is `tqdm` over saved episodes; discards are printed and summarized at the end.

## Output `.npz`

Each file is `np.savez_compressed` at `{output_dir}/episode_{index:04d}.npz`. Arrays are time-major (`T` = number of env steps in the episode).

| Key | Shape / type | Contents |
| --- | --- | --- |
| `times` | `(T,)` float64 | `arange(T) * step_seconds` |
| `actions` | `(T, 16)` float64 | Joint position targets in `LEAP_JOINT_ORDER` |
| `rewards` | `(T,)` float32 | `SqueezeCyringeReward` per step |
| `flex_dist/{flex_name}` | `(T, n_vert, 3)` | Vertex displacements |
| `flex_force/{flex_name}` | `(T, n_vert, 3)` | Kelvin–Voigt vertex forces |
| `flex_taxel_fk/positions` | `(T, 368, 3)` | FK taxel positions |
| `flex_taxel_fk/rotations` | `(T, 368, …)` | FK taxel rotations |
| `flex_taxel_fk/forces_local` | `(T, 368, 3)` | Taxel forces in local frames |
| `flex_taxel_fk/forces_world` | `(T, 368, 3)` | Taxel forces in world |
| `flex_taxel_fk/positions_deformed` | `(T, 368, 3)` | Deformed taxel positions |
| `joint_names` | `(16,)` str | `LEAP_JOINT_ORDER` |
| `metadata_json` | 0-d bytes/str | JSON object (see below) |

Load metadata with `json.loads(str(npz["metadata_json"]))`.

### Metadata

Shared across episodes:

| Field | Meaning |
| --- | --- |
| `config_path` | Resolved YAML path |
| `motion` | `cyringe_two_phase` or `random` |
| `profile` | Dataclass dict of the policy params |
| `step_seconds` | Seconds per recorded step |
| `max_steps_per_episode` | Collector cap from `duration` |
| `max_duration_seconds` | YAML `collection.duration` |
| `num_envs` | Parallel env count |
| `n_taxels` | 368 |
| `joint_names` | Same 16 names as `joint_names` array |
| `ee_site`, `goal_mocap`, `traj_mocap_names` | IK helper names (cyringe policy) |

Per episode:

| Field | Meaning |
| --- | --- |
| `episode` | Saved index (`0000` …) |
| `seed` | Seed used for that reset |
| `num_steps` | `T` |
| `terminated_by_env` | True if the env reported done (vector: done this step; single: terminated/truncated) |
| `termination_causes` | List of cause strings (empty if the collector duration cap fired first) |
| `env_slot` | Vector slot id (`0` for single env) |
| `cyringe_spawn` | Resolved spawn dict (`flex`, `scale`, `offset`, `euler`, `sticky`, `friction`, `housing_friction`) |

## Example configs

Two-phase IK (`cyringe.yaml`): 100 episodes, 10 s cap, 4 vector envs, viewer requested (no-op while `num_envs > 1`), DR off in YAML, trajectory IK with 15 waypoints.

Random policy (`cyringe_random_policy.yaml`): same horizon and env count, DR on, `action_hold: 40`, discards `flex_exploded`, `sim_unstable`, and `cyringe_flying`.
