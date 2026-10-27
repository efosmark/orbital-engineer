from gui.controller.client_sync import EngineModel
from gui.gtk4 import GObject
from gui.model.ledger import LedgerModel
from gui.model.osd import OnScreenDisplayModel
from gui.model.cmap import ColorMapModel

class AppModel(GObject.GObject):
    
    engine:EngineModel
    ledger:LedgerModel
    
    cmap:ColorMapModel
    osd:OnScreenDisplayModel
    
    secondary_body = GObject.Property(type=object, default=None)
    follow_tracked_body = GObject.Property(type=bool, default=True)
    show_grid = GObject.Property(type=bool, default=True)
    show_focused_history = GObject.Property(type=bool, default=False)
    show_force_vectors = GObject.Property(type=bool, default=False)
    show_orbital_ellipse = GObject.Property(type=bool, default=True)
    show_magnifier = GObject.Property(type=bool, default=False)
    show_debug_info = GObject.Property(type=bool, default=True)
    show_focus_info = GObject.Property(type=bool, default=True)
    show_grav_field = GObject.Property(type=bool, default=False)
    
    width = GObject.Property(type=int, default=0)
    height = GObject.Property(type=int, default=0)
    
    frame_clock = GObject.Property(type=object)
    fps = GObject.Property(type=float)
        
    particle_colors = GObject.Property(type=object)
    particle_names = GObject.Property(type=object)
    
    hover_position = GObject.Property(type=object)
    hovered_over_particle = GObject.Property(type=object)
    dragging_particle = GObject.Property(type=object)
    
    camera_drag_enable = GObject.Property(type=bool, default=True)
        
    selected_particles = GObject.Property(type=object)
    drag_start = GObject.Property(type=object)
    drag_end = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.osd = OnScreenDisplayModel()
        self.cmap = ColorMapModel()
        
        self.particle_colors = {}
        self.particle_names = {}
        self.hover_position = (0, 0)
        self.hovered_over_particle = None
        self.fps = 1.0
