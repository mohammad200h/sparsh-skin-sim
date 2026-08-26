# Design: LeapFlexCyringeEnv

This document describes `env/leap_flex_cyringe_env.py`: a Gymnasium environment that drives the Leap+XELA flex-sensor scene with a spawned **cyringe** (housing + sliding shaft). Spawn pose and physics defaults match `demos/demo_cyringe.py`.

## Purpose

Train or evaluate policies that close the Leap hand around a cyringe using **joint position targets**, with observations from:

- Flex-skin vertex displacements
- Kelvin–Voigt estimated contact forces on those vertices
- Forward-kinematics (FK) taxel pack used by the hardware/FK visualizer path

The environment does **not** run the demo’s two-phase IK motion. Actions are raw 16-DoF Leap targets; the demo remains the reference for spawn geometry and solver settings.

## Related files

| Path | Role |
| --- | --- |
| `env/leap_flex_cyringe_env.py` | Env class, `CyringeSpawn`, config loader |
| `env/env_config.json` | Default physics and spawn values |
| `env/rewad.py` | Reward (`squeeze` fingertip–object contact) |
| `util/objects_util.py` (`add_cyringe`) | Attach `cyringe/cyringe.xml` into the scene spec |
| `demos/demo_cyringe.py` | Canonical spawn / solver / thumb-axial init |
| `env/leap_flex_env.py` | Sibling tetris env (same obs/action skeleton) |

## Gymnasium contract

### Action

Shape `(16,)`, `float32`, clipped to each actuator’s `ctrlrange`.

Order is `LEAP_JOINT_ORDER` from `util/fk_taxel_util.py`. Each joint `name` maps to actuator `{name}_act`. The vector is written to `data.ctrl` at those actuator indices.

### Observation

`Dict` with three keys:

| Key | Contents |
| --- | --- |
| `flex_dist` | `{flex_name: (n_vert, 3)}` vertex displacements |
| `flex_force` | `{flex_name: (n_vert, 3)}` estimated forces (same names) |
| `flex_taxel_fk` | FK pack: `positions`, `rotations`, `forces_local`, `forces_world`, `positions_deformed` (368 taxels) |

Forces come from `AllFlexForceEstimator` (window from config, `use_qvel=True`). Taxel forces are mapped with `flex_forces_to_taxel_forces`, then `compute_fk_taxels`.

### Reward, termination, truncation

- **Reward:** `Reward(motion_type)` (default `"squeeze"`). Squeeze counts how many of `{if, mf, rf, th}` fingertips are in contact with any cyringe geom, scaled by `0.1`.
- **Terminated:** `False` unless `joint_movement_threshold` is set. Then `True` when the L2 norm of the per-step joint-angle change drops below that value (radians).
- **Truncated:** always `False`.
- **Info on reset:** `{"cyringe_spawn": {...}}` (resolved spawn). Step info is `{}`.

There is no Gymnasium `render()`; `metadata["render_modes"]` is empty. Callers can attach a MuJoCo viewer to `env.model` / `env.data`.

## Scene compile

On construction:

1. Load `env_config.json` (or `config_path`).
2. Merge optional `cyringe=` dict over JSON `cyringe`.
3. Resolve `CyringeSpawn`.
4. `MjSpec.from_file(scene_xml)` — default scene is  
   `leapXELA_model/scene_mjx_cube_CoACD_mjx_flex_sensor.xml`.
5. `add_cyringe(...)` with the resolved spawn (same arguments as the demo).
6. Compile; set `model.opt.iterations` (default **50**) and `model.opt.tolerance = 0`.
7. Cache actuator ids, ctrl limits, flex names, force estimator, fingertip/cyringe geom maps.

The housing body name is typically `cyringe/housing`. The plunger actuator is stripped inside `add_cyringe` (`keep_actuator=False`), so shaft slide is passive under contact.

`reset()` does **not** recompile. Spawn is fixed for the env instance. `mj_resetData` restores qpos/qvel; ctrl is zeroed; `th_axl_act` is set to `th_axl_initial` (default **1.6**) in both ctrl and the linked joint qpos.

Optional reset `options`: `"qpos"` (full `data.qpos`) and `"ctrl"` (16-DoF action, clipped).

## Cyringe spawn (parity with the demo)

Demo logic:

```text
spawn_offset = offset + (delta_x, delta_y, delta_z)
spawn_euler  = (0, π/2, −π/2 + delta_rot_z)
add_cyringe(..., above_palm=(flex is None), flex_name=flex,
            scale, offset=spawn_offset, euler=spawn_euler, sticky)
```

`CyringeSpawn.from_config` applies the same additions. JSON `euler` is the **base** pose; `delta_rot_z` is added to yaw (third component).

| Field | Default | Meaning |
| --- | --- | --- |
| `flex` | `null` | Named flex to spawn above; `null` → palm pads (`above_palm=True`) |
| `scale` | `1.0` | Uniform mesh / inertial scale |
| `offset` | `[0.01, 0.05, -0.12]` | Metres, added to the flex/palm spawn centre |
| `delta_x/y/z` | `0` | Extra world offset, folded into resolved `offset` |
| `euler` | `[0, π/2, −π/2]` | Side-lay: cyringe +Z is the long axis; 90° about Y |
| `delta_rot_z` | `0` | Extra yaw about world Z [rad] |
| `sticky` | `false` | If true, remove housing freejoint (barrel fixed; shaft still slides) |

Resolved spawn (`cyringe_spawn` / `CyringeSpawn.as_dict`) stores **combined** offset and euler, not the raw deltas.

## Contact detection (reward)

Fingertips:

- Flex: `flex_{if,mf,rf,th}_tip`
- Rigid tip geoms: bodies `{if,mf,rf,th}_ds` whose geom names contain `"tip"`

Object side: **all geoms** on bodies named `housing` or sharing the `cyringe/` prefix (housing + shaft), matching `_cyringe_geom_ids` in the demo.

A contact counts if a fingertip flex or tip geom pairs with a cyringe geom. Keys order: `if`, `mf`, `rf`, `th`.

## Step loop

1. Clip action into `data.ctrl`.
2. `mj_step` `n_substeps` times (default **1**).
3. Compute reward from current contacts.
4. Build observation (force estimator update).
5. Evaluate termination; store joint angles for the next movement check.

## Configuration

Defaults: `env/env_config.json`. Constructor kwargs override JSON (`None` means “use file”).

Top-level keys: `n_substeps`, `solver_iterations`, `th_axl_initial`, `force_window`, `joint_movement_threshold`, `motion_type`, `cyringe`.

```python
from env import LeapFlexCyringeEnv

env = LeapFlexCyringeEnv()
env = LeapFlexCyringeEnv(cyringe={"sticky": True, "delta_rot_z": 0.1})
env = LeapFlexCyringeEnv(config_path="path/to/other.json", n_substeps=5)
```

## What this env does not do

Compared to `demo_cyringe.py`:

- No thumb IK (`dls` / `mink` / `pyroki`), no IK goal mocap, no trajectory mocaps
- No MF/RF close-phase ramps
- No live flex / FK Open3D windows
- No domain randomization (unlike `LeapFlexEnv` + tetris)

Those belong in a demo, a wrapper policy, or a later env extension.

## Public API

- `load_env_config(path=None)`
- `CyringeSpawn` / `CyringeSpawn.from_config`
- `LeapFlexCyringeEnv` — `cyringe_spawn`, `model`, `data`, `reset`, `step`, `close`

Exported from `env/__init__.py`.
