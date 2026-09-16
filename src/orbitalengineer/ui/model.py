from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from orbitalengineer.ui.client_sync import EngineModel
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

@dataclass
class Pinpoint:
    position:np.complex64|complex
    start:float
    until:float
    radius:float
    color:tuple[float,float,float] = (1.0, 1.0, 1.0)

@dataclass
class OSDMessage:
    message:str
    duration:float
    category:str|None = None
    start:float|None = None
    description:str|None = None
    fade_in:bool = True
    now:float = 0


class OnScreenDisplayModel(GObject.GObject):
    props:Any
    osd_message = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.props.osd_message = []
        self._current_message = None     

    def add_message(self, message:str, duration:float=1, category:str|None=None, desc:str|None=None):
        # Remove any messages of the same category (e.g. allow overwriting same type)
        fade_in = len([m for m in self.osd_message if m.start is not None]) == 0
        self.osd_message = [m for m in self.osd_message if m.category != category]
        self.osd_message.append(OSDMessage(message, duration, category, description=desc, fade_in=fade_in))

    def get_message(self) -> OSDMessage|None:
        self._current_message = None
        if len(self.osd_message) == 0:
            return
        
        now = time.monotonic()
        for m in [*self.osd_message]:
            if m.start is None:
                m.start = now
            if now - m.start >= m.duration:
                self.osd_message.remove(m)
            if m.start > now:
                continue
            self._current_message = m
            break

        if self._current_message is not None:
            self._current_message.now = now
        return self._current_message


class AppModel(GObject.GObject):
    props:Any
    
    osd:OnScreenDisplayModel
    engine:EngineModel
    
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
    
    cmap = GObject.Property(type=object, default=None)
    cmap_id = GObject.Property(type=int, default=0)
    
    frame_clock = GObject.Property(type=object)
    fps = GObject.Property(type=float)
        
    particle_colors = GObject.Property(type=object)
    particle_names = GObject.Property(type=object)
    
    hover_position = GObject.Property(type=object)
    hovered_over_particle = GObject.Property(type=object)
    dragging_particle = GObject.Property(type=object)
    dragging_particle_offset = GObject.Property(type=object)
    
    start_maximized = GObject.Property(type=bool, default=False)
    camera_drag_enable = GObject.Property(type=bool, default=True)
        
    selected_particles = GObject.Property(type=object)
    drag_start = GObject.Property(type=object)
    drag_end = GObject.Property(type=object)
    durations = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.osd = OnScreenDisplayModel()
        self.engine = EngineModel()
        
        self.props.particle_colors = {}
        self.props.particle_names = {}
        self.props.hover_position = (0, 0)
        self.props.hovered_over_particle = None
        self.props.durations = defaultdict(list)
        self.props.fps = 1.0
        self.props.cmap = None
        

    def cycle_color_map(self, reverse:bool=False):
        if not reverse:
            self.cmap_id = (self.cmap_id + 1) % len(CMAP_TYPES)
        else:
            self.cmap_id = (self.cmap_id - 1) % len(CMAP_TYPES)
        self.cmap = CMAP_TYPES[self.cmap_id]
        self.osd.add_message(f"Color map: {self.cmap}", duration=2.0)

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