# Demo Implementation Notes

This document explains the flex-sensing demo and documents the public APIs in
`objects.py`, `flex_util.py`, `flex_visualizer.py`, and `demo.py`.

## Files

| File | Role |
|------|------|
| `leapXELA_model/scene_mjx_cube_CoACD_mjx_flex_sensor.xml` | Scene: Leap hand with soft flex skins (no free object in XML) |
| `objects.py` | Add free bodies to the scene via `MjSpec` (cube / tetrominoes) |
| `flex_util.py` | Flex topology, displacements, Kelvin–Voigt force estimation |
| `flex_visualizer.py` | Live matplotlib plots for displacement / force |
| `demo.py` | Loads scene, spawns a tetris piece, runs MuJoCo + visualizer |

Run:

```bash
poetry run python3 demo.py
poetry run python3 demo.py --visualize-force
poetry run python3 demo.py --flex flex_uspa46_1 --scale 2.0
```

CLI flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--visualize-force` | off | Color plots by estimated Kelvin–Voigt force [N] |
| `--no-visualize-force` | — | Force displacement coloring |
| `--vmax` | adaptive | Fixed color-scale max (m or N depending on mode) |
| `--flex` | palm pads | Spawn above this flex (`mf_tip` or `flex_uspa46_1`) |
| `--scale` | `1.5` | Uniform scale of the tetris piece |

---

## Approach

### What a flex is in this model

Each tactile patch is a MuJoCo **flex**: a soft membrane whose vertices are
individual bodies. Every vertex has **three slide joints** (local \(x, y, z\)).
Connect equalities gently anchor each vertex to its home pose on the parent
link. When something presses the skin, those joints move away from rest.

Natural / rest configuration is MuJoCo’s `model.qpos0`. Deformation at a vertex is:

\[
u = q - q_0 \in \mathbb{R}^{3}
\]

(typically millimetres).

### Reading deformation

`flex_joint_displacements` reads \(u\) for every vertex of a named flex.
Velocities \(v = \dot u\) come from `data.qvel` (preferred) or from a short
finite-difference history when only displacements are available.

### Force from multiple frames (Kelvin–Voigt)

Contact force is **not** stored in the joint state. We approximate each taxel as
a damped spring:

\[
\hat f_t \approx K\, u_t + B\, v_t
\]

- \(u_t\): current displacement (one frame)
- \(v_t\): velocity from `qvel` or from \(\{u_{t-1}, u_t, \ldots\}\)
- \(K\): stiffness [N/m]
- \(B\): damping [N·s/m]

Defaults: \(K \approx 5\) (from vertex mass / anchor time-constant), \(B \approx 0.05\)
(model joint damping). For better accuracy, collect batches of \((u, v)\) and
label forces with `flex_vertex_contact_forces` (MuJoCo contact wrenches), then
fit \(K, B\) with `fit_kelvin_voigt` / `FlexForceEstimator.calibrate`.

### Scene objects

The shipped scene XML has the hand + flex skins only. Free objects are added at
runtime through `MjSpec`:

1. `mj.MjSpec.from_file(scene.xml)`
2. `add_tetris_part(...)` / `add_cube(...)` / `add_tetris_parts(...)`
3. `spec.compile()` → `MjModel`

Tetrominoes are rigid bodies made of **four box geoms**. Contact bitmasks match
the old cube defaults (`contype=1`, `conaffinity=2`) so pieces collide with flex
skins and the floor.

Spawn placement (when `pos` is omitted):

- default / `above_palm=True` → above all palm pads matching `uspa46`
- `flex_name="mf_tip"` (or `"flex_uspa46_1"`) → above that flex’s vertex cloud
- explicit `pos=(x, y, z)` otherwise

Auto-spawn uses mean \(xy\) of the matched flex vertices and
\(z = \max z_{\text{vert}} + \text{half\_extent}_z + \text{clearance}\).

### Visualization

Live plots show rest-pose vertex layouts (PCA-projected to 2D) colored by
displacement magnitude (or \(x/y/z\)), or by estimated force when
`--visualize-force` is set. Color limits default to an adaptive peak-hold
tracker; pass `--vmax` for a fixed scale. The all-flex view also shows a global
history strip and a status line (peak value, `ncon`).

### Demo runtime details

- Solver iterations are raised to **50** (the shipped MJX XML uses 5, which
  under-converges soft skins).
- Simulation is paced in **real time** so dragging the piece in the viewer is usable.
- Pause the MuJoCo viewer (Space) for careful placement.
- Demo defaults: T-piece, `scale=1.5`, yaw \(45^\circ\), above palm (or `--flex`).

---

## Data flow

```text
scene XML
   └─ MjSpec.from_file
         └─ add_tetris_part(..., flex_name?, scale?, euler?)
               └─ spec.compile() → MjModel / MjData
                     ├─ flex_joint_displacements / AllFlexForceEstimator.update
                     │     └─ u, v  →  f = K u + B v
                     └─ AllFlexLiveVisualizer.update
                           └─ scatter map + history + status
```

---

## `objects.py` — constants

| Symbol | Meaning |
|--------|---------|
| `DEFAULT_CUBE_*` | Legacy reorientation-cube size / mass / colour / contact bits |
| `PALM_FLEX_SUBSTRING` | `"uspa46"` — matches all palm flexes |
| `DEFAULT_TETRIS_BLOCK_HALF` | `0.012` m half-extent of one tetromino cell |
| `DEFAULT_TETRIS_BLOCK_MASS` | `0.02` kg per cell (scaled by `scale**3`) |
| `DEFAULT_TETRIS_CLEARANCE` | Gap above the flex surface when auto-spawning |
| `DEFAULT_TETRIS_SPACING` | Centre-to-centre gap when placing multiple pieces |
| `TETRIS_SHAPES` | Cell offsets for `I/O/T/L/J/S/Z` |
| `TETRIS_COLORS` | Per-shape RGBA |

---

## `objects.py` — spawn helpers

### `palm_sensor_center(model, data=None) -> ndarray (3,)`

World-frame mean of all palm (`uspa46`) flex vertex centres.

### `spawn_pose_above_flex(spec, *, flex_name=None, flex_substring=None, half_size=…, clearance=…) -> ndarray (3,)`

Temporarily compiles `spec`, reads flex vertices, returns spawn centre above
them. Pass either `flex_name` (exact; `flex_` prefix optional) or
`flex_substring` (e.g. `"uspa46"`).

### `spawn_pose_above_palm(spec, *, half_size=…, clearance=…) -> ndarray (3,)`

Convenience wrapper: `spawn_pose_above_flex(..., flex_substring="uspa46")`.

### `_resolve_spawn_pos(spec, *, pos, flex_name, above_palm, half_size, clearance)`

Internal priority: explicit `pos` → `flex_name` → palm (`above_palm`) → error.

---

## `objects.py` — adding bodies

### `add_cube(spec, *, name="cube", pos=None, quat=…, half_size=…, mass=…, rgba=…, contype=…, conaffinity=…, freejoint=True, above_palm=True, flex_name=None, clearance=…) -> MjsBody`

Adds a free (or welded) box to `spec.worldbody`. Same spawn rules as tetris.

### `add_tetris_part(spec, *, shape="T", name=None, pos=None, quat=None, euler=None, scale=1.0, block_half=…, block_mass=…, rgba=None, freejoint=True, above_palm=True, flex_name=None, clearance=…) -> MjsBody`

Adds one rigid tetromino (4 box geoms).

- `scale`: multiplies `block_half`; mass scales as `scale**3`
- `euler`: XYZ radians (mutually exclusive with `quat`)
- `flex_name`: spawn above that flex when `pos` is `None`
- body name defaults to `tetris_{shape}`; geoms are `{name}_block{i}`

### `add_tetris_parts(spec, n, *, shapes=None, name_prefix="tetris", pos=None, quat=None, euler=None, scale=1.0, flex_name=None, above_palm=True, spacing=…, seed=None) -> list[MjsBody]`

Adds `n` pieces. If `shapes` is `None`, picks randomly (optionally seeded).
Pieces are spaced along \(+x\) from the first spawn pose.

Example:

```python
spec = mj.MjSpec.from_file(SCENE_XML.as_posix())
piece = add_tetris_part(
    spec,
    shape="T",
    flex_name="uspa46_1",  # or "flex_uspa46_1"
    scale=1.5,
    euler=(0.0, 0.0, np.pi / 4),
)
model = spec.compile()
```

---

## `flex_util.py` — constants

### `DEFAULT_FLEX_STIFFNESS` (`5.0`)

Default Kelvin–Voigt stiffness [N/m], motivated by vertex mass \(5\times10^{-4}\) kg
and anchor `solref` time constant \(\sim 0.01\) s (\(k \approx m/\tau^2\)).

### `DEFAULT_FLEX_DAMPING` (`0.05`)

Default damping [N·s/m], matching generated flex joint damping. If you leave
this default on `FlexForceEstimator`, it replaces \(B\) with the mean
`model.dof_damping` of that flex’s joints.

---

## `flex_util.py` — flex identity & topology

### `flex_id(model, flex_name) -> int`

Resolves a flex name (e.g. `"flex_if_tip"`) to its MuJoCo flex id. Raises if missing.

### `list_flex_names(model) -> list[str]`

Returns all flex names in model order (18 in the demo scene: palm pads, finger
pads, fingertips).

### `flex_vertex_body_ids(model, flex_name) -> ndarray (n_vert,)`

Body id of each flex vertex, in flex vertex order.

### `flex_joint_ids(model, flex_name) -> ndarray (n_vert, n_jnt)`

Joint ids owned by those vertex bodies. In this hand model `n_jnt == 3`
(slides along body \(x,y,z\)). Errors if vertices have mismatched joint counts.

### `_joint_qpos_width(jnt_type) -> int`

Qpos width for a joint type: free=7, ball=4, slide/hinge=1. Used to assert that
flex joints are scalar.

---

## `flex_util.py` — deformation & rest pose

### `flex_joint_displacements(model, data, flex_name) -> ndarray (n_vert, 3)`

\[
u = q_{\text{joint}} - q_0
\]

for every flex vertex joint. This is the primary “flex value” signal.

### `_with_natural_pose(model, data, fn)`

Temporarily sets `data.qpos = qpos0`, `qvel = 0`, runs forward kinematics,
calls `fn()`, then restores the previous state. Used to sample rest geometry
without permanently resetting the sim.

### `flex_rest_vertex_positions(model, data, flex_name) -> ndarray (n_vert, 3)`

World-frame vertex positions at the natural pose (via `_with_natural_pose`).

### `all_flex_rest_vertex_positions(model, data, flex_names=None) -> dict[str, ndarray]`

Same as above for many flexes in **one** natural-pose forward pass.

### `flex_joint_qpos_dof_addresses(model, flex_name) -> (qadr, dadr, shape)`

Cached index arrays into `qpos` / `qvel` for all joints of a flex, plus
`(n_vert, n_jnt)` shape. Used by the force estimator for fast updates.

### `flex_joint_velocities(model, data, flex_name) -> ndarray (n_vert, 3)`

Joint velocities from `data.qvel` for the flex’s slide DOFs.

---

## `flex_util.py` — force estimation

### `_as_coeff(coeff, shape) -> ndarray`

Broadcasts a scalar, per-axis `(3,)`, or per-vertex coefficient to full
`(n_vert, 3)` (or general `shape`).

### `kelvin_voigt_force(displacement, velocity, stiffness=…, damping=…) -> ndarray`

Pure function \(f = K \odot u + B \odot v\). Units: Newtons, body-local axes.

### `fit_kelvin_voigt(displacements, velocities, forces, *, per_axis=True) -> (K, B)`

Least-squares fit of \(f = K u + B v\) on batches shaped `(T, n_vert, 3)`.

- `per_axis=True`: independent \(K,B\) per \(x/y/z\) → arrays shape `(3,)`
- `per_axis=False`: one shared \(K,B\) for all components

### `flex_vertex_contact_forces(model, data, flex_name, contact_geom_names=None) -> (forces, count)`

**Ground-truth** labels for calibration:

1. Find MuJoCo contacts involving this flex (optionally only vs named geoms).
2. Read wrenches with `mj_contactForce`.
3. Splat onto vertices (exact vertex, or barycentric on the hit triangle).
4. Project world forces into each vertex body’s local axes (joint frame).

Returns `(n_vert, 3)` forces [N] and the number of contacts used.

For a tetris piece, pass its block geom names, e.g.
`["tetris_T_block0", "tetris_T_block1", "tetris_T_block2", "tetris_T_block3"]`.

### class `FlexForceEstimator`

Online multi-frame estimator for **one** flex.

| Method / property | Meaning |
|-------------------|---------|
| `__init__(model, flex_name, stiffness=…, damping=…, window=5, use_qvel=True, velocity_smoothing=0.3)` | Cache addresses; keep a displacement history of length `window`. |
| `shape` | `(n_vert, 3)` |
| `reset()` | Clear history and smoothed velocity. |
| `set_gains(stiffness, damping)` | Install \(K, B\) without fitting. |
| `calibrate(U, V, F, per_axis=True)` | `fit_kelvin_voigt` then `set_gains`. |
| `update(model, data) -> (force, u, v)` | Push current frame; return estimated force [N], displacement [m], velocity [m/s]. |
| `history_displacements() -> (T, n_vert, 3)` | Recent \(u\) buffer. |

### class `AllFlexForceEstimator`

Wraps one `FlexForceEstimator` per flex.

| Method | Meaning |
|--------|---------|
| `__init__(model, flex_names=None, **kwargs)` | Build estimators for all (or selected) flexes. |
| `reset()` | Reset every child. |
| `update(model, data) -> dict[str, ndarray]` | `{flex_name: force(n_vert, 3)}` |
| `peak_force_magnitude(forces) -> float` | Max per-vertex \(\|f\|\) across all flexes [N]. |

---

## `flex_visualizer.py` — plotting helpers

### `_CHANNEL_INDEX`

Maps plot channel names to joint-axis indices: `"x"→0`, `"y"→1`, `"z"→2`,
`"magnitude"→None` (Euclidean norm).

### `_project_to_plane(points) -> ndarray (n, 2)`

PCA projection of 3D rest positions onto their best-fit plane.

### `_channel_values(values, channel) -> ndarray (n_vert,)`

Extracts \(x\), \(y\), \(z\), or magnitude for coloring (works for \(u\) or \(f\)).

### `ClimTracker` / `_adaptive_clim`

Peak-hold adaptive color limits when `vmax` is `None`.

### `_short_flex_label(flex_name) -> str`

Strips the `"flex_"` prefix for compact subplot titles.

### `_grid_shape(n) -> (rows, cols)`

Near-square subplot grid for \(n\) flexes.

---

## `flex_visualizer.py` — visualizers

### class `FlexLiveVisualizer`

Live plot for **one** flex (spatial map + history). Supports `visualize_force`.

### `visualize_flex_live(model, data, flex_name, **kwargs) -> FlexLiveVisualizer`

Factory for `FlexLiveVisualizer`.

### class `AllFlexLiveVisualizer`

Live grid for **all** (or selected) flexes.

| Method | Meaning |
|--------|---------|
| `__init__(..., channel="magnitude", vmax=None, update_hz=20, visualize_force=False)` | One scatter per flex + shared colorbar + global history + status. |
| `update(model, data, forces=None, force=False)` | Refresh panels. When `visualize_force=True`, pass `forces={flex_name: (n_vert,3)}`. |
| `close()` | Close the figure. |

### `visualize_all_flexes_live(model, data, flex_names=None, **kwargs) -> AllFlexLiveVisualizer`

Factory for `AllFlexLiveVisualizer`.

---

## `demo.py`

### `SCENE_XML`

Path to `leapXELA_model/scene_mjx_cube_CoACD_mjx_flex_sensor.xml`.

### `SOLVER_ITERATIONS` (`50`)

Override for `model.opt.iterations` so flex contacts converge (XML ships with 5).

### `parse_args()`

CLI: `--visualize-force`, `--no-visualize-force`, `--vmax`, `--flex`, `--scale`.

### `main()`

1. Parse CLI flags.
2. Load the scene into `MjSpec`.
3. `add_tetris_part` (T-piece, yaw \(45^\circ\), `--scale`, optional `--flex`).
4. Compile → set solver iterations / tolerance → `MjData`.
5. Print model size, flex names, spawn info.
6. Open `visualize_all_flexes_live` (`visualize_force` from CLI).
7. Create `AllFlexForceEstimator(window=5, use_qvel=True)`.
8. Loop while the passive MuJoCo viewer runs:
   - `mj_step`
   - `viewer.sync`
   - `force_est.update` → forces
   - `viz.update(..., forces=forces)` when force mode is on
   - sleep to keep wall-clock ≈ sim timestep (real-time pacing)
9. Close the matplotlib window on exit.

---

## Suggested calibration snippet

```python
from flex_util import (
    FlexForceEstimator,
    flex_vertex_contact_forces,
)

piece_geoms = [f"tetris_T_block{i}" for i in range(4)]
est = FlexForceEstimator(model, "flex_uspa46_1", window=5)
Us, Vs, Fs = [], [], []

for _ in range(500):
    mj.mj_step(model, data)
    f_hat, u, v = est.update(model, data)
    f_gt, _ = flex_vertex_contact_forces(
        model, data, "flex_uspa46_1", piece_geoms
    )
    Us.append(u)
    Vs.append(v)
    Fs.append(f_gt)

import numpy as np
K, B = est.calibrate(np.stack(Us), np.stack(Vs), np.stack(Fs))
print("fitted K [N/m]:", K, "B [N·s/m]:", B)
```

Typical fitted stiffness after a palm press is on the order of **50–100 N/m**
(higher than the conservative default of 5).

---

## Practical tips

1. **Spawn target:** use `--flex flex_uspa46_1` (or `mf_tip`) to drop the piece
   on a specific pad; omit `--flex` for the whole palm.
2. **Scale:** `--scale 1.0` is the base cell size; demo default is `1.5`.
3. **Lift vs press:** patches light up under contact and go dark when you lift.
4. **Pause the viewer** before dragging free bodies; real-time pacing helps but
   pause is still easier for precise placement.
5. **Calibrate** if you care about absolute Newtons; defaults are order-of-magnitude only.
6. Prefer **`use_qvel=True`**; finite differences on millimetre-scale \(u\) are noisy.
7. Keep **solver iterations ≥ ~50** when collecting force / displacement data.
