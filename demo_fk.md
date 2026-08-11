# `demo_fk.py` — Flex sensor demo with FK taxel visualization

Interactive demo that loads the Leap+XELA hand with soft flex skins, optionally
drops a test object on a chosen taxel, and runs **two** live views:

1. **MuJoCo viewer** — physics + matplotlib flex heatmaps (displacement or force)
2. **Open3D FK panel** — taxel positions and forces in FK layout (368 taxels)

Use this demo to check that simulated contact forces appear on the **same grid
cell** in MuJoCo and in the FK visualizer.

---

## Quick start

```bash
# Tetris piece above palm (default)
poetry run python3 demo_fk.py

# Heavy cube on one palm taxel
poetry run python3 demo_fk.py --flex uspa46_1 --taxel-row 1 --taxel-col 2

# Middle-finger proximal pad, centre cell
poetry run python3 demo_fk.py --flex mf_px_uspa44 --taxel-row 1 --taxel-col 1

# Middle fingertip
poetry run python3 demo_fk.py --flex mf_tip --taxel-row 2 --taxel-col 2
```

---

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--flex` | *(palm)* | Target flex name (`mf_tip`, `uspa46_2`, `flex_mf_px_uspa44`, …) |
| `--taxel-row` | — | FK grid row (0-indexed). Requires `--flex` and `--taxel-col` |
| `--taxel-col` | — | FK grid column (0-indexed). Requires `--flex` and `--taxel-row` |
| `--taxel-mass` | `2.0` | Cube mass in kg (taxel-cube mode only) |
| `--scale` | `1.5` | Tetris piece scale (tetris mode only) |
| `--offset DX DY DZ` | `0 0 0` | Extra world-frame spawn offset (metres) |
| `--visualize-force` | off | Colour MuJoCo flex plots by estimated force [N] |
| `--no-visualize-force` | — | Force displacement colouring (default in tetris mode) |
| `--vmax` | adaptive | Fixed colour-scale max (m or N) |
| `--no-fk-viz` | off | Disable Open3D FK window |

### Two spawn modes

**Taxel cube** — set both `--taxel-row` and `--taxel-col` (and `--flex`):

- Spawns a **small heavy cube** (2.5 mm, 2.0 kg) above that taxel
- Force visualization is **auto-enabled** so you can compare with Open3D
- Startup prints `vertex=` and `taxel=` for the resolved cell

**Tetris piece** — omit `--taxel-row` / `--taxel-col`:

- Spawns a **T-tetromino** above the palm (default) or above `--flex`
- Pause MuJoCo (space) and drag the piece onto a pad to probe contact

The `flex_` prefix is optional: `mf_tip` and `flex_mf_tip` are equivalent.

---

## Grid coordinates

`--taxel-row` and `--taxel-col` use the **FK / Open3D panel layout** — the same
labelling as `leap_sensor_taxel_map.json` after FK reordering (`util/fk_taxel_util.py`).

Indices are **zero-based**. Row increases downward on the panel; column increases
to the right (matching the blue fingertip / red palm grids in Open3D).

### Grid sizes by patch type

| Patch type | Rows × cols | Example flex names |
|------------|-------------|-------------------|
| Palm `uspa46` | 4 × 6 | `uspa46_1`, `uspa46_2`, `uspa46_3` |
| Finger `uspa44` | 4 × 4 | `mf_bs_uspa44`, `mf_px_uspa44`, `mf_md_uspa44`, … |
| Fingertip | 6 × 6 *(sparse)* | `mf_tip`, `if_tip`, `rf_tip`, `th_tip` |

Fingertip pads only have **30** active taxels; many cells on the 6×6 canvas are
empty. Picking an empty cell raises an error.

---

## Flex name cheat sheet

### Palm (4×6 each)

| `--flex` | Location |
|----------|----------|
| `uspa46_1` | Palm pad (up-left patch) |
| `uspa46_2` | Palm pad (up-right patch) |
| `uspa46_3` | Palm pad (down-left patch) |

**Note on finger 4×4 pads:** FK local ``yy`` is flipped for all `*4x4*` URDF
links so the IF/RF sides match MuJoCo (otherwise a cube on the right of a pad
lights the left corner in Open3D).

**Note on `uspa46_2`:** This palm sensor is mounted **180° in-plane** relative to
the other palm pads. FK positions for this patch are mirrored in
`util.fk_taxel_util.get_fk_taxel_frames` so the Open3D layout matches the MuJoCo flex
mesh. Grid `(row, col)` still follows the FK panel — no extra flags needed.

### Middle finger (MF), palm → tip

| `--flex` | Segment | Grid |
|----------|---------|------|
| `mf_bs_uspa44` | Base (near palm) | 4×4 |
| `mf_px_uspa44` | Proximal phalanx | 4×4 |
| `mf_md_uspa44` | Middle phalanx | 4×4 |
| `mf_tip` | Fingertip | 6×6 |

### Other fingers

Same naming pattern:

- **Index (IF):** `if_bs_uspa44`, `if_px_uspa44`, `if_md_uspa44`, `if_tip`
- **Ring (RF):** `rf_bs_uspa44`, `rf_px_uspa44`, `rf_md_uspa44`, `rf_tip`
- **Thumb (TH):** `th_px_uspa44`, `th_ds_uspa44`, `th_tip`

On IF/MF/RF, the geom suffixes `px` and `md` do not match JSON patch names
`second` / `third`; spawn resolution uses the correct taxel→vertex map in
`flex_vertex_for_fk_grid`.

---

## MF fingertip 6×6 canvas

Taxel IDs per cell (`.` = empty):

```
row\col   0    1    2    3    4    5
  0     185  204  205  206  207  190
  1     203  221  222  223  224  208
  2     164  156  246  247  248  225
  3     186  165  157  158  159  249
  4       .  187  166  167  168    .
  5       .    .  188  189    .    .
```

Example — centre of the active region:

```bash
poetry run python3 demo_fk.py --flex mf_tip --taxel-row 2 --taxel-col 2
# → taxel 246
```

---

## Validation workflow

1. Pick a flex and grid cell, e.g. `--flex uspa46_1 --taxel-row 0 --taxel-col 0`
2. Run `demo_fk.py` and let the cube settle on the pad
3. In **MuJoCo**, check the flex heatmap (force mode) at the contact location
4. In **Open3D**, check the highlighted taxel on the matching palm/finger patch
5. Compare startup line: `grid=(row, col) … taxel=NNN` with the hot cell in Open3D

If they disagree, note the printed `taxel=` id and grid coordinates — that is the
ground truth for debugging FK vs simulation alignment.

---

## Open3D FK window

Opened unless `--no-fk-viz` is set.

| Key | Action |
|-----|--------|
| **F** | Toggle deformed vs rest taxel positions |
| **V** | Toggle force vector arrows |
| **Q** | Quit Open3D (MuJoCo keeps running until closed separately) |

Forces come from `flex_forces_to_taxel_forces` (Kelvin–Voigt estimate per flex
vertex) passed through `compute_fk_taxels`.

---

## MuJoCo flex plots

Matplotlib windows show per-flex heatmaps updated at ~15 Hz:

- **Default (tetris mode):** vertex **displacement** magnitude
- **Taxel-cube / `--visualize-force`:** estimated **force** magnitude [N]

Use `--vmax` to fix the colour scale when comparing runs.

---

## How taxel spawn works (implementation)

1. `add_cube_on_taxel` in `util/objects_util.py` compiles the scene spec temporarily
2. `flex_vertex_for_fk_grid` in `util/fk_taxel_util.py` maps `(flex, row, col)` →
   hardware taxel id → MuJoCo flex vertex index
3. Cube centre is placed above that vertex along the taxel outward normal
   (+ clearance, + optional `--offset`)

Defaults: cube half-size **1.25 mm** (2.5 mm side), mass **2.0 kg**, clearance **60 mm**.

Programmatic use:

```python
import mujoco as mj
from util.objects_util import add_cube_on_taxel

spec = mj.MjSpec.from_file("leapXELA_model/scene_mjx_cube_CoACD_mjx_flex_sensor.xml")
add_cube_on_taxel(spec, flex_name="mf_px_uspa44", grid_row=1, grid_col=1)
model = spec.compile()
```

---

## Related files

| File | Role |
|------|------|
| `demo_fk.py` | This demo (MuJoCo + FK Open3D) |
| `demo.py` | Same spawn options, no FK visualizer |
| `util/objects_util.py` | `add_cube_on_taxel`, `spawn_pose_above_taxel`, tetris helpers |
| `util/fk_taxel_util.py` | FK taxel frames, grid→vertex mapping, Open3D visualizer |
| `util/flex_util.py` | Kelvin–Voigt force estimation |
| `util/flex_visualizer.py` | Live MuJoCo flex heatmaps |
| `demo_implementation.md` | Deeper API notes for util.flex_util / util.objects_util / demo.py |

Scene XML:

`leapXELA_model/scene_mjx_cube_CoACD_mjx_flex_sensor.xml`

---

## Example commands

```bash
# Palm corner cell
poetry run python3 demo_fk.py --flex uspa46_2 --taxel-row 0 --taxel-col 0

# Sweep a row on a 4×4 finger pad
poetry run python3 demo_fk.py --flex mf_md_uspa44 --taxel-row 2 --taxel-col 0
poetry run python3 demo_fk.py --flex mf_md_uspa44 --taxel-row 2 --taxel-col 3

# Heavier cube, slight vertical offset
poetry run python3 demo_fk.py --flex mf_tip --taxel-row 3 --taxel-col 3 \
  --taxel-mass 1.0 --offset 0 0 0.005

# MuJoCo only (no Open3D)
poetry run python3 demo_fk.py --flex uspa46_1 --taxel-row 1 --taxel-col 1 --no-fk-viz

# Classic tetris drop on middle fingertip
poetry run python3 demo_fk.py --flex mf_tip --scale 1.2
```
