# Cyringe IK report — `dls`

## Config

- IK backend: `dls`
- sticky: `True`
- close_duration: `1.0` s
- delta_ik_z: `0.02` m
- reach_tol: `0.008` m
- duration: `3.0` s

## Tracking error (thumb EE → IK goal)

| metric | value |
| --- | --- |
| IK steps | 2001 |
| mean ‖e‖ [m] | 0.039817 |
| min ‖e‖ [m] | 0.029050 |
| final ‖e‖ [m] | 0.029050 |
| mean |dx| [m] | 0.012336 |
| mean |dy| [m] | 0.032036 |
| mean |dz| [m] | 0.015806 |
| time to reach [s] | nan |
