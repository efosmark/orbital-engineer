from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol, cast

from orbitalengineer.engine import config
from orbitalengineer.ui.gtk4 import Gtk, Gdk, Graphene, GObject, GLib
import numpy as np

@dataclass
class Pinpoint:
    position:np.complex64|complex
    start:float
    until:float
    radius:float
    color:tuple[float,float,float] = (1.0, 1.0, 1.0)

class DataModel(GObject.GObject):
    secondary_body = GObject.Property(type=object, default=None)

    props:Any

    # Time sampling
    render = GObject.Property(type=object)
    frame = GObject.Property(type=object)
    bump = GObject.Property(type=object)

    num_steps = GObject.Property(type=int, default=0)
    num_ticks = GObject.Property(type=int, default=0)
    steps_per_tick = GObject.Property(type=object)
            
    def __init__(self):
        super().__init__()
        self.durations = defaultdict(list)


class ViewModel(GObject.GObject):
    props:Any
    
    paused = GObject.Property(type=bool, default=True)
    speed = GObject.Property(type=float, default=config.DEFAULT_SPEED)
    
    fps = GObject.Property(type=object)
    
    start_maximized = GObject.Property(type=bool, default=False)
    show_plot_at_startup = GObject.Property(type=bool, default=True)
    
    follow_tracked_body = GObject.Property(type=bool, default=True)
    show_grid = GObject.Property(type=bool, default=True)
    show_focused_history = GObject.Property(type=bool, default=False)
    show_all_history = GObject.Property(type=bool, default=False)
    show_force_vectors = GObject.Property(type=bool, default=False)
    show_orbital_ellipse = GObject.Property(type=bool, default=True)
    show_reticle = GObject.Property(type=bool, default=True)
    show_magnifier = GObject.Property(type=bool, default=False)
    show_debug_info = GObject.Property(type=bool, default=True)
    show_points = GObject.Property(type=bool, default=True)
    show_focus_info = GObject.Property(type=bool, default=True)
    
    pinpoint = GObject.Property(type=object)
    particle_colors = GObject.Property(type=object)
    particle_names = GObject.Property(type=object)
    
    hover_position = GObject.Property(type=object)
    hovered_over_particle = GObject.Property(type=object)
    dragging_particle = GObject.Property(type=object)
    dragging_particle_offset = GObject.Property(type=object)
    
    camera_drag_enable = GObject.Property(type=bool, default=True)
    
    width = GObject.Property(type=int, default=0)
    height = GObject.Property(type=int, default=0)
    
    font_family = GObject.Property(type=str, default="Liberation Mono")
    font_size = GObject.Property(type=int, default=10)
    
    osd_message = GObject.Property(type=object)    
    
    selected_particles = GObject.Property(type=object)
    drag_start = GObject.Property(type=object)
    drag_end = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.props.pinpoint = []
        self.props.particle_colors = {}
        self.props.particle_names = {}
        self.props.hover_position = (0, 0)
        self.props.hovered_over_particle = None
        self.props.osd_message = []
    
