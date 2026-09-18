import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from orbitalengineer.engine import logger
from orbitalengineer.ui.model.main import AppModel
from orbitalengineer.ui.model.cmap import OPT_KE, OPT_MASS, OPT_MOMENTUM


class ColorizedMapController:
    original_colors:dict|None = None
    colors:dict|None = None
    last_time:float|None = None

    def __init__(self, app:AppModel):
        self.app = app
        self.model = app.cmap
        
        self.model.connect('notify::option', self.on_color_map_changed)
        self.model.connect('notify::colormap', self.on_color_map_changed)
        self.model.connect('notify::gamma', self.on_color_map_changed)
        self.app.engine.connect('notify::tick-id', self.on_color_map_changed)
    
    def on_color_map_changed(self, _model, param):
        if self.model.option is None or self.model.colormap is None:
            if self.original_colors is not None:
                self.app.props.particle_colors = self.original_colors
                self.original_colors = None
            return
        
        if self.original_colors is None:
            self.original_colors = dict(self.app.props.particle_colors)
        if self.colors is None:
            self.colors = dict()
        
        self.last_option_type = self.model.option
                
        if self.model.option == OPT_KE:
            values = [
               float((0.5 * self.app.engine.mass[i] * (abs(self.app.engine.velocity[i])**2))) #+ (self.app.engine.mass[i] * (300_000**2)))
               for i in range(self.app.engine.N)
            ]
        elif self.model.option == OPT_MOMENTUM:
            values = [
                self.app.engine.mass[i] * abs(self.app.engine.velocity[i])
                for i in range(self.app.engine.N)
            ]
        elif self.model.option == OPT_MASS:
            values = [ self.app.engine.mass[i] for i in range(self.app.engine.N) ]
        else:
            logger.warning('Unknown color map option: %s', self.model.option)
            return
        
        vmin, vmax = np.nanpercentile(values, [1, 99])
        norm = colors.PowerNorm(gamma=self.model.gamma, vmin=vmin, vmax=vmax, clip=False)
        cmap = plt.colormaps[self.model.colormap]
        
        for b in self.app.engine.valid_indices:
            self.colors[b] = cmap(norm(values[b]) * 0.95 + 0.05)
            self.app.props.particle_colors[b] = self.colors[b]
