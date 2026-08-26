# Cyringe IK report — `mink`

## Config

- IK backend: `mink`
- sticky: `True`
- close_duration: `1.0` s
- delta_ik_z: `0.02` m
- reach_tol: `0.008` m
- duration: `4.0` s

## Tracking error (thumb EE → IK goal)

| metric | value |
| --- | --- |
| IK steps | 3001 |
| mean ‖e‖ [m] | 0.037509 |
| min ‖e‖ [m] | 0.025689 |
| final ‖e‖ [m] | 0.025742 |
| mean |dx| [m] | 0.018186 |
| mean |dy| [m] | 0.013167 |
| mean |dz| [m] | 0.025552 |
| time to reach [s] | nan |
