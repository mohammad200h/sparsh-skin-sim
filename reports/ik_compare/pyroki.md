# Cyringe IK report — `pyroki`

## Config

- IK backend: `pyroki`
- sticky: `True`
- close_duration: `1.0` s
- delta_ik_z: `0.02` m
- reach_tol: `0.008` m
- duration: `3.0` s

## Tracking error (thumb EE → IK goal)

| metric | value |
| --- | --- |
| IK steps | 2001 |
| mean ‖e‖ [m] | 0.036512 |
| min ‖e‖ [m] | 0.022681 |
| final ‖e‖ [m] | 0.022681 |
| mean |dx| [m] | 0.011321 |
| mean |dy| [m] | 0.032894 |
| mean |dz| [m] | 0.007677 |
| time to reach [s] | nan |
