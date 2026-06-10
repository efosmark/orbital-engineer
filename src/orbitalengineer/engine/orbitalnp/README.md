# Orbital N-Body Simulation (NumPy/Numba backed)

An N-body orbital engine that implements a Leapfrog KDK integrator using JIT-compiled Numba kernels.
Supports body merging or collision bouncing.

**NOTE** - This is not functional anymore. It was the primary engine prior to migrating to OpenCL.

## Theory of Operation

 1. Spool up a number of worker processes (typically 1 fewer than the number of processors)
 2. Partition the bodies currently being simulated such that each worker gets roughly the same count
 3. Use shared memory & limit ranges to the worker processes. Do not allow writing to memory
    used by other workers.
 4. The "host" process is in charge of syncronization via barriers and MP events

## Structure

 |    file            | purpose                                                          |
 | -------------------|----------------------------------------------------------------- |
 | `orbitalnp.py`     | Initializes, controls, and dispatches workers. Processes ticks.  |
 | `sshm.py`          | A structured shared memory handler (used by workers).            |
 | `orbitalmemory.py` | Implements the `sshm` for the orbital state parameters.          |
 | `worker.py`        | Worker process. Handles a specific partition of bodies.          |
 | `integrator/*.py`  | Various Numba-compiled kernels used by the integrator.           |

## Problems

- The orbitalnp engine is out-of-date compared to orbitalcl
  - It references incorrect class names
  - It does not use the `SimClock` (instead handled internally)
  - It does not match the structure used by orbitalcl
- It does not support the `flags` property
- It does not properly implement swept detection, causes small fast-moving particles to have
  undefined behavior
- Merge strategy has to be handled globally
- It does not emit integrator metrics to the metrics app

## Plan

- Once `orbitalcl` is complete, this will be refreshed to match the structure
- Enable the app to choose which integrator to use
