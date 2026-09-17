from dataclasses import dataclass
from orbitalengineer.ui.gtk4 import GObject

COLORMAPS:list[str] = ['hot', 'afmhot', 'plasma', 'gnuplot2', 'viridis', 'cividis', 'gist_rainbow']

@dataclass
class CMapOption:
    name:str
    default_colormap:str
    default_gamma:float = 1.0

OPT_KE = CMapOption('Kinetic Energy', 'afmhot')
OPT_MOMENTUM = CMapOption('Momentum', 'gnuplot2')
OPT_MASS = CMapOption('Mass', 'viridis')

CMAP_OPTIONS = [
    None,
    OPT_KE,
    OPT_MOMENTUM,
    OPT_MASS,
]

class ColorMapModel(GObject.GObject):
    
    option:CMapOption|None = GObject.Property(type=object, default=None) #type:ignore
    option_id = GObject.Property(type=int, default=0)
    gamma = GObject.Property(type=float, default=1.0)
    colormap = GObject.Property(type=object)
    colormap_id = GObject.Property(type=int, default=0)
    
    def cycle(self, reverse:bool=False):
        if not reverse:
            self.option_id = (self.option_id + 1) % len(CMAP_OPTIONS)
        else:
            self.option_id = (self.option_id - 1) % len(CMAP_OPTIONS)
        self.option = CMAP_OPTIONS[self.option_id]
        if self.option is not None:
            self.gamma = self.option.default_gamma
            self.colormap = self.option.default_colormap
            self.colormap_id = COLORMAPS.index(self.colormap)
        self.notify('option')
    
    def increase_gamma(self):
        self.gamma /= 0.9
    
    def decrease_gamma(self):
        self.gamma *= 0.9
        
    def prev_colormap(self):
        self.colormap_id = (self.colormap_id - 1) % len(COLORMAPS)
        self.colormap = COLORMAPS[self.colormap_id]
        self.notify('colormap')
        
    def next_colormap(self):
        self.colormap_id = (self.colormap_id + 1) % len(COLORMAPS)
        self.colormap = COLORMAPS[self.colormap_id]
        self.notify('colormap')