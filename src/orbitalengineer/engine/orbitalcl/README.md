# Orbital N-Body Simulation (OpenCL backed)

An N-body orbital engine that implements a Leapfrog KDK integrator using OpenCL
kernels. Supports body merging or collision bouncing.

## Structure

### Top-level Files

| file             | purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| `orbitalcl.py`   | Host contoller responsible for dispatching CL kernels    |
| `particle_cl.py` | Proxy object for fetching field values for a particle    |
| `tracer.py`      | EventTracer emits kernel metrics                         |
| `flags.py`       | Bitwise per-body feature flags                           |
| `pipeline_step`  | Class `PipelineStep`                                     |
| `device.py`      | Misc tools for analyzing CL-compatible devices           |
| `kernel/*`       | Inclusion helpers (.clh files)                           |

### Kernels

| kernel          | purpose                                                   |
| --------------- | --------------------------------------------------------- |
| `interaction/*` | Time-of-impact between bodies                             |
| `position/*`    | Position based on dt and state vectors                    |
| `velocity/*`    | Velocity from half of the dt                              |
| `bounce/*`      | Flips impulse for bodies with a `BOUNCE` flag on collide  |
| `merge/*`       | Combines bodies with a `MERGE` flag. Compute based on CoM |
| `nudge/*`       | Optionally resolve any overlapping bodies during startup  |

## Pipeline Overview

    ╭───╮
    ╰─╥─╯
      ║                                       ━┓
    ╭─╨─────────────────────────────────╮      ┃ SWEPT DETECTION
    │ compute_interaction_time          │      ┃
    ├───────────────────────────────────┤      ┃
    │                                   │      ┃
    │  IN:               OUT:           │      ┃
    │  - position        - toi_dt       │      ┃
    │  - velocity        - node_min_dt  │      ┃
    │  - radius                         │      ┃
    │                                   │      ┃
    ╰─╥─────────────────────────────────╯      ┃
      ║                                       ━┛
      ║                                       ━┓
    ╭─╨──────────────────────────────╮         ┃ KICK
    │ compute_velocity               │         ┃
    ├────────────────────────────────┤         ┃ 
    │                                │         ┃ 
    │  IN:               OUT:        │         ┃ 
    │  - node_min_dt     - velocity  │         ┃ 
    │  - position                    │         ┃ 
    │  - mass                        │         ┃ 
    │  - radius                      │         ┃ 
    │                                │         ┃ 
    ╰─╥──────────────────────────────╯         ┃
      ║                                       ━┛
      ║                                       ━┓
    ╭─╨──────────────────────────────╮         ┃ DRIFT
    │ compute_position               │         ┃ 
    ├────────────────────────────────┤         ┃
    │                                │         ┃
    │  IN:               OUT:        │         ┃
    │  - node_min_dt     - position  │         ┃
    │  - velocity                    │         ┃
    │                                │         ┃
    ╰─╥──────────────────────────────╯         ┃
      ║                                       ━┛  
      ║                                       ━┓
    ╭─╨──────────────────────────────╮         ┃ KICK
    │ compute_velocity               │         ┃
    ├────────────────────────────────┤         ┃ 
    │                                │         ┃ 
    │  IN:               OUT:        │         ┃ 
    │  - node_min_dt     - velocity  │         ┃ 
    │  - position                    │         ┃ 
    │  - mass                        │         ┃ 
    │  - radius                      │         ┃ 
    │                                │         ┃ 
    ╰─╥──────────────────────────────╯         ┃
      ║                                       ━┛
      ║                                       ━┓
    ╭─╨──────────────────────────────────╮     ┃ COLLISION DETECTION
    │ compute_relative_velocity          │     ┃ 
    ├────────────────────────────────────┤     ┃ 
    │                                    │     ┃ 
    │  IN:               OUT:            │     ┃ 
    │  - position        - rel_velocity  │     ┃ 
    │  - velocity                        │     ┃ 
    │                                    │     ┃ 
    ╰─╥──────────────────────────────────╯     ┃ 
      ║                                        ┃       
    ╭─╨──────────────────────────────╮         ┃ 
    │ collision_group_assign         │         ┃
    ├────────────────────────────────┤         ┃
    │                                │         ┃
    │  IN:             OUT:          │         ┃
    │  - node_min_dt   - coll_group  │         ┃
    │  - flags                       │         ┃ 
    │  - mass                        │         ┃
    │  - rel_velocity                │         ┃
    │  - radius                      │         ┃
    │                                │         ┃
    ╰─╥──────────────────────────────╯         ┃
      ║                                        ┃ 
    ╭─╨────────────────────────────╮           ┃ 
    │ compute_merging_collision    │           ┃ 
    ├──────────────────────────────┤           ┃ 
    │                              │           ┃ 
    │  IN:             OUT:        │           ┃ 
    │  - flags         - flags     │           ┃ 
    │  - coll_group    - position  │           ┃ 
    │  - position      - velocity  │           ┃ 
    │  - velocity      - mass      │           ┃ 
    │  - mass          - radius    │           ┃ 
    │  - radius                    │           ┃ 
    │                              │           ┃ 
    ╰─╥────────────────────────────╯           ┃ 
      ║                                        ┃
    ╭─╨──────────────────────────────────╮     ┃  
    │ apply_bounce                       │     ┃  
    ├────────────────────────────────────┤     ┃  
    │                                    │     ┃  
    │  IN:               OUT:            │     ┃  
    │  - flags           - velocity      │     ┃  
    │  - position        - position      │     ┃  
    │  - velocity        - bounce_point  │     ┃  
    │  - mass                            │     ┃  
    │                                    │     ┃  
    ╰─╥──────────────────────────────────╯     ┃
      ║                                       ━┛
    ╭─╨─╮
    ╰───╯
