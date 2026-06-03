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


class ViewModel(GObject.GObject):
    props:Any
    
    paused = GObject.Property(type=bool, default=True)
    speed = GObject.Property(type=float, default=config.DEFAULT_SPEED)
    
    secondary_body = GObject.Property(type=object, default=None)
    follow_tracked_body = GObject.Property(type=bool, default=True)
    show_grid = GObject.Property(type=bool, default=True)
    show_focused_history = GObject.Property(type=bool, default=False)
    show_all_history = GObject.Property(type=bool, default=False)
    show_force_vectors = GObject.Property(type=bool, default=False)
    show_orbital_ellipse = GObject.Property(type=bool, default=True)
    show_magnifier = GObject.Property(type=bool, default=False)
    show_debug_info = GObject.Property(type=bool, default=True)
    show_focus_info = GObject.Property(type=bool, default=True)
    
    fps = GObject.Property(type=object)
        
    pinpoint = GObject.Property(type=object)
    particle_colors = GObject.Property(type=object)
    particle_names = GObject.Property(type=object)
    
    hover_position = GObject.Property(type=object)
    hovered_over_particle = GObject.Property(type=object)
    dragging_particle = GObject.Property(type=object)
    dragging_particle_offset = GObject.Property(type=object)
    
    start_maximized = GObject.Property(type=bool, default=False)
    camera_drag_enable = GObject.Property(type=bool, default=True)
    
    width = GObject.Property(type=int, default=0)
    height = GObject.Property(type=int, default=0)
    
    font_family = GObject.Property(type=str, default="Liberation Mono")
    font_size = GObject.Property(type=int, default=10)
    
    osd_message = GObject.Property(type=object)    
    
    selected_particles = GObject.Property(type=object)
    drag_start = GObject.Property(type=object)
    drag_end = GObject.Property(type=object)
    durations = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.props.pinpoint = []
        self.props.particle_colors = {}
        self.props.particle_names = {}
        self.props.hover_position = (0, 0)
        self.props.hovered_over_particle = None
        self.props.osd_message = []
        self.props.durations = defaultdict(list)

    def to_dict(self) -> dict:
        return {
            "paused": self.props.paused,
            "speed": self.props.speed,
            "secondary_body": int(self.props.secondary_body) if self.props.secondary_body != None else None,
            "follow_tracked_body": self.props.follow_tracked_body,
            "show_grid": self.props.show_grid,
            "show_focused_history": self.props.show_focused_history,
            "show_all_history": self.props.show_all_history,
            "show_force_vectors": self.props.show_force_vectors,
            "show_orbital_ellipse": self.props.show_orbital_ellipse,
            "show_magnifier": self.props.show_magnifier,
            "show_debug_info": self.props.show_debug_info,
            "show_focus_info": self.props.show_focus_info,
            "particle_names": self.props.particle_names,
            "particle_colors": self.props.particle_colors
        }

    def load_from_dict(self, obj:dict):
        self.props.paused = obj["paused"]
        self.props.speed = obj["speed"]
        self.props.secondary_body = obj["secondary_body"]
        self.props.follow_tracked_body = obj["follow_tracked_body"]
        self.props.show_grid = obj["show_grid"]
        self.props.show_focused_history = obj["show_focused_history"]
        self.props.show_all_history = obj["show_all_history"]
        self.props.show_force_vectors = obj["show_force_vectors"]
        self.props.show_orbital_ellipse = obj["show_orbital_ellipse"]
        self.props.show_magnifier = obj["show_magnifier"]
        self.props.show_debug_info = obj["show_debug_info"]
        self.props.show_focus_info = obj["show_focus_info"]
        self.props.particle_names = dict([(int(k), v) for k, v in obj["particle_names"].items()])
        self.props.particle_colors = dict([(int(k), v) for k, v in obj["particle_colors"].items()])