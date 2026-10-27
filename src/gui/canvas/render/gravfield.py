"""This works, but needs to be refactored. Should be kernel-backed.
Also needs to have computations performed outside of the rendering code.
"""

import math
import time
import cairo
from orbitalengineer.engine.orbitalnp.integrator.accel import force
from gui.canvas import renderer
import numpy as np
from numpy.typing import NDArray

from numba import njit
import matplotlib.pyplot as plt

cmap = plt.colormaps['cubehelix']

@njit
def fast_lognorm(vals, vmin, vmax):
    vals = np.maximum(vals, 1e-12)
    logv = np.log(vals)
    scale = 1.0 / (np.log(vmax) - np.log(vmin))
    return (logv - np.log(vmin)) * scale

@njit
def min_nonzero(vals:NDArray):
    result = vals[0]
    for val in vals:
        if result == 0:
            result = val
            continue
        if 0 < val < result:
            result = val
    return result

@njit
def is_bounded(position:complex, x_origin:float, x_max:float, y_origin:float, y_max:float):
    max_dist = max(y_max-y_origin, x_max-x_origin)
    return position.real >= x_origin - max_dist \
       and position.real <= x_max + max_dist \
       and position.imag >= y_origin - max_dist \
       and position.imag <= y_max + max_dist

@njit
def compute_forces(forces, N, num_cols, num_rows, x_origin, y_origin, x_max, y_max, resolution, position, mass):
    MAX_DIST = max(y_max-y_origin, x_max-x_origin) * 2
    max_dist_per_particle = np.empty(N, dtype=float)
    max_dist_per_particle.fill(0)
    
    ii = 0
    ix = np.empty((N,), dtype=np.int64)
    for i in range(N):
        if not is_bounded(position[i], x_origin, x_max, y_origin, y_max):
            continue
        ix[ii] = i
        ii += 1
    ix = ix[:ii]
    
    for row in range(num_rows):
        y_pos = (row * resolution) + (resolution/2.0)
        
        for col in range(num_cols):
            x_pos = (col * resolution) + (resolution/2.0)
            
            for i in ix:
                dr =  position[i] - complex(x_origin + x_pos, y_origin + y_pos)
                dist = np.abs(dr)
                if dist > MAX_DIST:continue
                forces[col, row] += mass[i] / (dist**3)
    return forces

RESOLUTION = 10

class GravFieldRenderer(renderer.Renderer):

    #def do_draw(self, snapshot, width, height):
    #    snapshot.push_blur(RESOLUTION)
    #    super().do_draw(snapshot, width, height)
    #    snapshot.pop()
    
    def initialize(self):
        self.forces = None
        self.forces_min = 0
        self.forces_max = 0
        self.last_computed_at = 0

    def draw(self, cr:cairo.Context, width:int, height:int):
        if not self.app.show_grav_field or self.app.engine.N == 0: return
        N = self.app.engine.N
        num_cols = math.ceil(width / RESOLUTION)
        num_rows = math.ceil(height / RESOLUTION)
        x_origin, y_origin = self.camera.screen_to_world(0, 0, width, height)
        x_max, y_max = self.camera.screen_to_world(width, height, width, height)
        mass = self.app.engine.mass
        position = self.app.engine.position
        
        now = time.monotonic()
        if self.forces is None or now - self.last_computed_at > 0.25:
            self.last_computed_at = now
            self.forces = np.empty((num_cols, num_rows), dtype=np.float64)
            self.forces.fill(0)
            
            compute_forces(self.forces, N, num_cols, num_rows, x_origin, y_origin, x_max, y_max, RESOLUTION/self.camera.zoom, position, mass)
            self.forces_min = min_nonzero(self.forces.flatten())/2.0
            self.forces_max = max((np.abs(force(np.float32(1), mass[i], 1, np.float32(1.0))) for i in range(self.app.engine.N)))
        
        y_offset = 0
        x_offset = 0
        try:
            for row in range(num_rows):
                for col in range(num_cols):
                    if self.forces[col, row] > 0:
                        cr.set_source_rgb(*cmap(fast_lognorm(self.forces[col, row], self.forces_min, self.forces_max))[:3])
                        cr.rectangle(x_offset, y_offset, RESOLUTION, RESOLUTION)
                        cr.fill()
                    
                    x_offset += RESOLUTION
                x_offset = 0
                y_offset += RESOLUTION
        except IndexError:
            ... # Probably a resize. Wait for next computation.
