from collections import deque
import time
import cairo
from orbitalengineer.engine import logger
from orbitalengineer.ui.canvas import renderer
from matplotlib import colors

import matplotlib.pyplot as plt

from orbitalengineer.ui.model import CMAP_DIST, CMAP_KE, CMAP_MASS, CMAP_MV
cmap_afmhot = plt.colormaps['afmhot']
cmap_gist_rainbow = plt.colormaps['gist_rainbow']
cmap_plasma = plt.colormaps['plasma']
cmap_viridis = plt.colormaps['viridis']

class MomentumColorizedRenderer(renderer.Renderer):
    original_colors:dict|None = None
    max_hist = None
    min_hist = None
    last_tick = None
    colors:dict|None = None
    last_time:float|None = None
    last_cmap_type:str|None = None

    def draw(self, cr:cairo.Context, width:int, height:int):
        if self.app.cmap is None:
            if self.original_colors is not None:
                self.app.props.particle_colors = self.original_colors
                self.original_colors = None
            return
        
        if self.original_colors is None:
            self.original_colors = dict(self.app.props.particle_colors)
        if self.colors is None:
            self.colors = dict()
        if self.last_time is None or self.app.cmap != self.last_cmap_type:
            self.last_time = time.monotonic()
        if self.max_hist is None or self.app.cmap != self.last_cmap_type:
            self.max_hist = deque(maxlen=100)
        if self.min_hist is None or self.app.cmap != self.last_cmap_type:
            self.min_hist = deque(maxlen=100)
        
        
        self.last_cmap_type = self.app.cmap
        
        # We don't need to recompute every frame
        t = time.monotonic()
        frame_rate = self.app.fps or 10.0
        target_frame_rate = frame_rate * 0.25
        if t - self.last_time < (1.0/target_frame_rate):
            return
        self.last_time = t
        
        if self.app.cmap == CMAP_KE:
            values = [
               float((0.5 * self.orbital.mass[i] * (abs(self.orbital.velocity[i])**2))) #+ (self.orbital.mass[i] * (300_000**2)))
               for i in range(self.orbital.N)
            ]
            cmap = cmap_afmhot
        elif self.app.cmap == CMAP_MV:
            values = [
                self.orbital.mass[i] * abs(self.orbital.velocity[i])
                for i in range(self.orbital.N)
            ]
            cmap = cmap_plasma
        elif self.app.cmap == CMAP_MASS:
            values = [ self.orbital.mass[i] for i in range(self.orbital.N) ]
            cmap = cmap_viridis
        elif self.app.cmap == CMAP_DIST:
            values = [ abs(self.orbital.position[i]) for i in range(self.orbital.N) ]
            cmap = cmap_gist_rainbow
        else:
            logger.warning('Unknown color map option: %s', self.app.cmap)
            return
        
        if self.last_tick is None or self.orbital.tick_id > self.last_tick:
            self.last_tick = self.orbital.tick_id
            try:
                self.max_hist.append(max(values))
                self.min_hist.append(min(values))
            except ValueError:
                return
        
        
        try:
            vmin = sum(self.min_hist)/len(self.min_hist)
            vmax = sum(self.max_hist)/len(self.max_hist)
        except ZeroDivisionError:
            vmin = min(values)
            vmax = max(values)
        
        #norm = colors.PowerNorm(1.0, vmin=vmin, vmax=vmax, clip=True)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        
        for b in self.orbital:
            if b.idx is None: continue
            self.colors[b.idx] = cmap(norm(values[b.idx]) * 0.95 + 0.05)
            self.app.props.particle_colors[b.idx] = self.colors[b.idx]