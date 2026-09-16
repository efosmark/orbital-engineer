from typing import Any

from orbitalengineer.engine import config
from orbitalengineer.ipc.client import ClientSocketConnection
from orbitalengineer.ui.gtk4 import GObject, GLib

SYNC_RATE_MS = int((1/30.0) * 1000.0)

_TENTHS = [v/10.0 for v in range(10)]
SPEED_SCALE = [
    *[(v+1)/100.0 for v in range(9)],
    *[v for v in _TENTHS if v > 0],
    *[(1 + v) for v in _TENTHS],
    *[(2 + v) for v in _TENTHS],
    *[(3 + v) for v in _TENTHS],
    *[(4 + v) for v in _TENTHS],
    *[(5 + v) for v in _TENTHS],
    *[(6 + v) for v in _TENTHS],
    *[(7 + v) for v in _TENTHS],
    *[(8 + v) for v in _TENTHS],
    *[(9 + v) for v in _TENTHS],
]

class EngineModel(GObject.GObject): 
    props:Any
       
    tick_id = GObject.Property(type=int, default=0)
    dt_step = GObject.Property(type=float, default=0)
    is_initialized = GObject.Property(type=bool, default=True)
    N = GObject.Property(type=int, default=0)
    accum = GObject.Property(type=float, default=0)
    valid_indices = GObject.Property(type=object)
    
    paused = GObject.Property(type=bool, default=True)
    clock_speed = GObject.Property(type=float, default=config.DEFAULT_SPEED)
    clock_time = GObject.Property(type=float, default=0.0)
    max_speed = GObject.Property(type=object)
    curr_tick_at = GObject.Property(type=float, default=0.0)
    next_tick_at = GObject.Property(type=float, default=0.0)

    def increase_speed(self):
        idx = SPEED_SCALE.index(self.clock_speed) + 1
        if idx >= len(SPEED_SCALE):
            idx = len(SPEED_SCALE) - 1
        self.clock_speed = SPEED_SCALE[idx]

    def decrease_speed(self):
        idx = SPEED_SCALE.index(self.clock_speed) - 1
        if idx < 0:
            idx = 0
        self.clock_speed = SPEED_SCALE[idx]



class ClientSyncController(GObject.GObject):
    props:Any
    
    def __init__(self, client:ClientSocketConnection, model:EngineModel):
        super().__init__()
        self.client = client
        self.model = model
        self.valid_indices = []
        
        self.model.connect('notify::clock-speed', self.on_clock_speed_changed)
        GLib.timeout_add(SYNC_RATE_MS, self.sync)
        
    def sync(self):
        self.client.sync()

        if self.model.clock_speed != self.client.clock.speed:
            self.model.clock_speed = self.client.clock.speed
        self.model.clock_time = self.client.clock.time()
        self.model.tick_id = self.client.tick_id
        self.model.dt_step = self.client.dt_step
        self.model.is_initialized = self.client.is_initialized
        self.model.N = self.client.N
        self.model.accum = self.client.accum

        self.model.max_speed = self.client.max_speed
        self.model.curr_tick_at = self.client.curr_tick_at
        self.model.next_tick_at = self.client.next_tick_at
        self.model.valid_indices = self.client.get_valid_indices()
        self.model.notify('valid-indices')
        
        return True
    
    def on_clock_speed_changed(self, model, param):
        print('on_clock_speed_changed', self.model.clock_speed)
        if not self.model.is_initialized: return
        self.client.set_clock_speed(self.model.clock_speed)
        #return True