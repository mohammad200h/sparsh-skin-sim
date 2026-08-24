# Demos

Run from the repo root with Poetry.

## Cyringe thumb IK (`demo_cyringe.py`)

Two-phase motion: MF/RF close, then thumb IK toward the cyringe handle.
`--sticky` fixes the housing so the shaft can still slide.

IK backends (`--ik`): `dls` (custom DLS), `mink`, `pyroki`.

### Viewer (live goal = handle + `--delta-ik-z`)

```bash
poetry run python demos/demo_cyringe.py --sticky --ik mink --delta-ik-z 0.02

poetry run python demos/demo_cyringe.py --sticky --ik pyroki --delta-ik-z 0.02
```

### Precomputed trajectory (recommended for squeeze)

IK follows a push path along handle local +Z. Blue→red mocap spheres mark
waypoints; the yellow `ik_goal` mocap is the current waypoint.

Defaults: stroke **0.08 m**, **15** waypoints (same spacing as the old 0.04 m / 8 wp path).

```bash
poetry run python demos/demo_cyringe.py --sticky --ik mink --trajectory

poetry run python demos/demo_cyringe.py --sticky --ik pyroki --trajectory
```

Explicit length / waypoint count:

```bash
poetry run python demos/demo_cyringe.py --sticky --ik mink \
  --trajectory --traj-stroke 0.08 --traj-waypoints 15

poetry run python demos/demo_cyringe.py --sticky --ik pyroki \
  --trajectory --traj-stroke 0.08 --traj-waypoints 15
```

### Headless + XYZ error log

```bash
poetry run python demos/demo_cyringe.py --sticky --ik mink --headless --duration 3 \
  --trajectory --log-ik reports/ik_compare/mink.csv --report reports/ik_compare/mink.md

poetry run python demos/demo_cyringe.py --sticky --ik pyroki --headless --duration 3 \
  --trajectory --log-ik reports/ik_compare/pyroki.csv --report reports/ik_compare/pyroki.md
```

Per-step CSV columns: `time`, `phase`, `dx`, `dy`, `dz`, `dist`.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--ik dls\|mink\|pyroki` | Thumb IK solver |
| `--sticky` | Fix housing in world |
| `--trajectory` | Follow precomputed handle-+Z path |
| `--offset DX DY DZ` | Base spawn offset [m] (default `0.01 0.05 -0.12`) |
| `--delta-x` `--delta-y` `--delta-z` | Extra world XYZ offset [m] (default 0) |
| `--delta-rot-z` | Extra yaw about world Z [rad] (default 0) |
| `--traj-stroke` | Path length [m] (default 0.08) |
| `--traj-waypoints` | Number of mocap waypoints (default 15) |
| `--delta-ik-z` | Live lead along handle +Z (no `--trajectory`) |
| `--no-motion` | Viewer only |
| `--headless` | No windows |
| `--duration` | Stop after N sim seconds |
| `--log-ik` / `--report` | CSV + markdown tracking report |

Comparison write-up: `reports/ik_compare/comparison.md`.
