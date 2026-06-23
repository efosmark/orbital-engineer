# Orbital N-Body Simulation (OpenCL backed)

An N-body orbital engine that implements a Leapfrog KDK integrator using OpenCL
kernels. Supports body merging or collision bouncing.

## Theory of Operation

Fundamentally, the engine is split into a pipeline of several OpenCL kernels. The controller is
executed on the host, and is responsible for initialization and kernel dispatching. The simulation
is performed tick-by-tick, using [leapfrog KDK integration](https://en.wikipedia.org/wiki/Leapfrog_integration)
in order to continually compute orbital state vectors for velocity and position.

Each tick is broken down into three phases:

 1. **Swept detection** - Compute the time-of-impact for each pair of bodies and track the minimum dt.
    If the minimum dt is smaller than the base dt for the tick, it is used as the input dt for
    future steps.
     - By ensuring the dt is never larger than the earliest collision, the engine avoids
    having a body overlap or 'tunnel' through another body.

 2. **Leapfrog KDK** - Integrate position and velocity via three steps:

    |   | step       | dt input       | updates vector    |
    | - | ---------- | -------------- | ----------------- |
    | 1 | `KICK`     | $\frac{dt}{2}$ | velocity          |
    | 2 | `DRIFT`    | $dt$           | position          |
    | 3 | `KICK`     | $\frac{dt}{2}$ | velocity          |

 3. **Collision Detection** - Determine which bodies are touching, and apply collision operations.
    - **_None_** : Bodies pass through each-other. In order to avoid asymptotes, overlapping bodies
      do not continue to impart force until they are no longer overlapping. This is the default
      strategy for bodies that do not have a `MERGE` or `BOUNCE` flag.
    - **_Merge_** : Bodies are combined. Their velocities and positions are computed based on
      their center-of-mass.
    - **_Bounce_** : Bodies deflect off of eachother. The strength is controlled by the
      [coefficient of restitution](https://en.wikipedia.org/wiki/Coefficient_of_restitution).
      Set via the config option, `COEF_OF_RESTITUTION`.

## Structure

### Top-level Files

| file                  | purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `orbitalcl.py`        | Host contoller responsible for dispatching CL kernels.                   |
| `particle_cl.py`      | Proxy object for fetching field values for a particle.                   |
| `tracer.py`           | EventTracer emits kernel metrics.                                        |
| `flags.py`            | Bitwise per-body feature flags. Builds `flags.clh` for kernel usage.     |
| `pipeline_step.py`    | Class `PipelineStep`                                                     |
| `device.py`           | Misc tools for analyzing CL-compatible devices. Helpful for enumerating. |
| `kernel/*`            | Inclusion helpers (.clh files).                                          |

### Kernels

| kernel                | purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `interaction/*`       | Kernels responsible for computing time-of-impact between each body.      |
| `position/*`          | Computes position based on the tick dt and orbital state vectors.        |
| `velocity/*`          | Computes velocity from half of the dt.                                   |
| `edge_distance/*`     | Computes the edge-to-edge particle distance (dist - r1 - r2)             |
| `relative_velocity/*` | Computes `v_rel` for each pair of bodies.                                |
| `bounce/*`            | Updates vectors for bodies with a `BOUNCE` flag. Flips impulse.          |
| `merge/*`             | Combines bodies with a `MERGE` flag. Compute based on CoM.               |
| `nudge/*`             | Optionally resolve any overlapping bodies.                               |

## Pipeline Overview

    ╭───╮
    ╰─╥─╯
      ║
    ╭─╨─────────────────────────────────╮     ━┓
    │ compute_interaction_time          │      ┃ SWEPT DETECTION
    ├───────────────────────────────────┤      ┃
    │                                   │      ┃
    │  IN:               OUT:           │      ┃
    │  - position        - toi_dt       │      ┃
    │  - velocity        - node_min_dt  │      ┃
    │  - radius                         │      ┃
    │                                   │      ┃
    ╰─╥─────────────────────────────────╯     ━┛
      ║
    ╭─╨──────────────────────────────╮        ━┓
    │ compute_velocity               │         ┃ KICK
    ├────────────────────────────────┤         ┃ 
    │                                │         ┃ 
    │  IN:               OUT:        │         ┃  
    │  - node_min_dt     - velocity  │         ┃ 
    │  - position                    │         ┃ 
    │  - mass                        │         ┃ 
    │  - radius                      │         ┃ 
    │  - edge_distance               │         ┃ 
    │                                │         ┃ 
    ╰─╥──────────────────────────────╯        ━┛
      ║                                         
    ╭─╨──────────────────────────────╮        ━┓
    │ compute_position               │         ┃ DRIFT
    ├────────────────────────────────┤         ┃
    │                                │         ┃
    │  IN:               OUT:        │         ┃
    │  - node_min_dt     - position  │         ┃
    │  - velocity                    │         ┃
    │                                │         ┃
    ╰─╥──────────────────────────────╯         ┃ 
      ║                                        ┃
    ╭─╨───────────────────────────────────╮    ┃
    │ compute_edge_distance               │    ┃
    ├─────────────────────────────────────┤    ┃
    │                                     │    ┃
    │  IN:               OUT:             │    ┃
    │  - position        - edge_distance  │    ┃
    │  - radius                           │    ┃
    │                                     │    ┃
    ╰─╥───────────────────────────────────╯   ━┛
      ║                                         
    ╭─╨──────────────────────────────╮        ━┓
    │ compute_velocity               │         ┃ KICK
    ├────────────────────────────────┤         ┃ 
    │                                │         ┃ 
    │  IN:               OUT:        │         ┃ 
    │  - node_min_dt     - velocity  │         ┃ 
    │  - position                    │         ┃ 
    │  - mass                        │         ┃ 
    │  - radius                      │         ┃ 
    │  - edge_distance               │         ┃ 
    │                                │         ┃ 
    ╰─╥──────────────────────────────╯        ━┛
      ║                                         
    ╭─╨──────────────────────────────────╮    ━┓ 
    │ compute_relative_velocity          │     ┃ COLLISION DETECTION
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
    │  - edge_distance               │         ┃
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
    │  - rel_velocity                    │     ┃  
    │  - edge_distance                   │     ┃  
    │                                    │     ┃  
    ╰─╥──────────────────────────────────╯    ━┛  
      ║
    ╭─╨─╮
    ╰───╯
