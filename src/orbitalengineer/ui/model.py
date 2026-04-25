from collections import defaultdict, deque
from dataclasses import dataclass
import time
from typing import Any, Protocol, cast

from orbitalengineer.ui.gtk4 import Gtk, Gdk, Graphene, GObject, GLib
import numpy as np

@dataclass
class Pinpoint:
    position:np.complex128
    start:float
    until:float
    radius:float

class OSDMessage:
    message:str
    start:float
    until:float|None
    def __init__(self, message:str, duration:float|None):
        self.message = message
        self.start = time.monotonic()
        self.until = self.start + duration if duration is not None else None

class _ViewModelProps(Protocol):
    pinpoint: Any
    particle_colors: Any
    particle_names: Any
    hover_position: Any
    hovered_over_particle: Any
    osd_message: Any

class DataModel(GObject.GObject):
    secondary_body = GObject.Property(type=object, default=None)

    props:Any

    # Time sampling
    render = GObject.Property(type=object)
    frame = GObject.Property(type=object)
    bump = GObject.Property(type=object)
    durations = GObject.Property(type=object)

    num_steps = GObject.Property(type=int, default=0)
    num_ticks = GObject.Property(type=int, default=0)
    steps_per_tick = GObject.Property(type=object)
    
    def add_duration(self, name, duration, tick_id):
        self.durations[name].append((tick_id, duration))
        while len(self.durations[name]) > 0 and self.durations[name][0][0]  < self.num_ticks - 300:
            self.durations[name].pop(0)
        
    def __init__(self):
        super().__init__()
        self.durations = defaultdict(list)


class ViewModel(GObject.GObject):
    props:Any
    
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
    
    width = GObject.Property(type=int, default=0)
    height = GObject.Property(type=int, default=0)
    
    font_family = GObject.Property(type=str, default="Liberation Mono")
    font_size = GObject.Property(type=int, default=10)
    
    osd_message = GObject.Property(type=object)
    
    
    def __init__(self):
        super().__init__()
        props = cast(_ViewModelProps, self.props)
        props.pinpoint = []
        props.particle_colors = {}
        props.particle_names = {}
        props.hover_position = (0, 0)
        props.hovered_over_particle = None
        props.osd_message = []
    
    def add_osd_message(self, message, duration:float|None=None) -> OSDMessage:
        m = OSDMessage(message, duration)
        props = cast(_ViewModelProps, self.props)
        props.osd_message.append(m)
        return m
        
