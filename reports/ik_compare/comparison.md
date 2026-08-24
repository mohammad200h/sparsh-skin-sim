# Cyringe thumb IK comparison

Headless sticky runs, `duration=3s`, `delta_ik_z=0.02`, same MF/RF close then thumb IK.

Error is world-frame `goal − EE` (thumb flex-center site → handle site + δz).

## Backends

- **dls**: custom damped least-squares Jacobian IK (`util/ik_dls_util.py`)
- **mink**: [Mink](https://kevinzakka.github.io/mink/) QP differential IK on a reduced thumb MJCF (`util/ik_mink_util.py`)
- **pyroki**: [PyRoki](https://pyroki-toolkit.github.io/) JAX LM IK on a thumb URDF (`util/ik_pyroki_util.py`)

## Summary

| backend | IK steps | mean ‖e‖ [m] | min ‖e‖ [m] | final ‖e‖ [m] | mean |dx| | mean |dy| | mean |dz| | t_reach [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dls` | 2001 | 0.039817 | 0.029050 | 0.029050 | 0.012336 | 0.032036 | 0.015806 | nan |
| `mink` | 2001 | 0.038388 | 0.025817 | 0.025817 | 0.023932 | 0.025233 | 0.005881 | nan |
| `pyroki` | 2001 | 0.036512 | 0.022681 | 0.022681 | 0.011321 | 0.032894 | 0.007677 | nan |

## Per-backend reports

- [dls.md](dls.md) / [dls.csv](dls.csv)
- [mink.md](mink.md) / [mink.csv](mink.csv)
- [pyroki.md](pyroki.md) / [pyroki.csv](pyroki.csv)

## Usage

```bash
poetry run python demos/demo_cyringe.py --sticky --ik dls|mink|pyroki \
  --headless --duration 3 --delta-ik-z 0.02 \
  --log-ik reports/ik_compare/run.csv --report reports/ik_compare/run.md
```
