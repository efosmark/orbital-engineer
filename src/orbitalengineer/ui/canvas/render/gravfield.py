import math
import cairo
from orbitalengineer.engine.orbitalnp.integrator.accel import force
from orbitalengineer.ui.canvas import renderer
from orbitalengineer.ui.gtk4 import Gtk
import numpy as np
from numpy.typing import NDArray

from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import colors

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
        #max_dist_per_particle[i] = np.cbrt((mass[i]/1e-5))
        ii += 1
    ix = ix[:ii]
    
    for row in range(num_rows):
        y_pos = (row * resolution) + (resolution/2.0)
        
        for col in range(num_cols):
            x_pos = (col * resolution) + (resolution/2.0)
            
            for i in ix:
                
                dr =  position[i] - complex(x_origin + x_pos, y_origin + y_pos)
                dist = np.abs(dr)
                
                if dist > MAX_DIST:# or (dist > max_dist_per_particle[i] and max_dist_per_particle[i] > 0):
                    continue
                
                forces[col, row] += mass[i] / (dist**3)
                
                # F = M / dist**3
                # F/M = dist**3
                # cbrt(F/M) = dist
    return forces

RESOLUTION = 20

class GravFieldRenderer(renderer.Renderer):

    def do_draw(self, snapshot, width, height):
        snapshot.push_blur(RESOLUTION)
        super().do_draw(snapshot, width, height)
        snapshot.pop()

    def draw(self, cr:cairo.Context, width:int, height:int):
        N = self.orbital.N
        num_cols = math.ceil(width / RESOLUTION)
        num_rows = math.ceil(height / RESOLUTION)
        x_origin, y_origin = self.camera.screen_to_world(0, 0, width, height)
        x_max, y_max = self.camera.screen_to_world(width, height, width, height)
        mass = self.orbital.mass
        position = self.orbital.position
        
        #max_dist = max(y_max-y_origin, x_max-x_origin)/10
        forces = np.empty((num_cols, num_rows), dtype=np.float64)
        forces.fill(0)
        compute_forces(forces, N, num_cols, num_rows, x_origin, y_origin, x_max, y_max, RESOLUTION/self.camera.zoom, position, mass)
        
        forces_min = min_nonzero(forces.flatten())/2.0
        #forces_min = min((np.abs(force(1, mass[i], RESOLUTION, 1.0)) for i in range(self.orbital.shm.N)))
        #forces_max = forces.max()
        forces_max = max((np.abs(force(np.float32(1), mass[i], 1, np.float32(1.0))) for i in range(self.orbital.N)))
        
        
        y_offset = 0
        x_offset = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if forces[col, row] > 0:
                    cr.set_source_rgb(*cmap(fast_lognorm(forces[col, row], forces_min, forces_max))[:3])
                    cr.rectangle(x_offset, y_offset, RESOLUTION, RESOLUTION)
                    cr.fill()
                
                x_offset += RESOLUTION
            x_offset = 0
            y_offset += RESOLUTION
        
