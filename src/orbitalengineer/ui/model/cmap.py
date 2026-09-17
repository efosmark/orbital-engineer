from typing import Any
from orbitalengineer.ui.gtk4 import GObject


CMAP_KE = 'Kinetic Energy'
CMAP_MV = 'Momentum'
CMAP_MASS = 'Mass'
CMAP_DIST = 'Distance'
CMAP_TYPES = [
    None,
    CMAP_KE,
    CMAP_MV,
    CMAP_MASS,
    #CMAP_DIST
]

class ColorMapModel(GObject.GObject):
    props:Any
    
    cmap = GObject.Property(type=object, default=None)
    cmap_id = GObject.Property(type=int, default=0)
    
    def __init__(self):
        super().__init__()
        self.props.cmap = None
    
    def cycle(self, reverse:bool=False):
        if not reverse:
            self.cmap_id = (self.cmap_id + 1) % len(CMAP_TYPES)
        else:
            self.cmap_id = (self.cmap_id - 1) % len(CMAP_TYPES)
        self.cmap = CMAP_TYPES[self.cmap_id]
        self.notify('cmap')