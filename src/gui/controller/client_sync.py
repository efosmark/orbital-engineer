from gui.ui_config import SYNC_RATE_MS
from gui.gtk4 import GObject, GLib
from orbitalengineer.ipc.client import ClientSocketConnection
from gui.model.engine import EngineModel

class ClientSyncController(GObject.GObject):
    
    def __init__(self, client:ClientSocketConnection, model:EngineModel):
        super().__init__()
        self.client = client
        
        self.model = model
        self.model.connect('notify::clock-speed', self.on_clock_speed_changed)
        GLib.timeout_add(SYNC_RATE_MS, self.sync)
        
    def sync(self):
        try:
            self.client.sync()
        except ConnectionResetError:
            raise SystemExit
        
        self.model.clock_time = self.client.clock.time()
        
        if self.model.is_initialized != self.client.is_initialized:
            self.model.is_initialized = self.client.is_initialized

        if not self.client.is_initialized:
            return
        
        if self.model.paused != (not self.client.clock.running):
           self.model.paused = not self.client.clock.running
        
        if self.model.clock_speed != self.client.clock.speed:
            self.model.clock_speed = self.client.clock.speed
        
        try:
            if self.model.config is None and self.client.config is not None:
                self.model.config = self.client.config
                self.model.notify('config')
        except AttributeError:
            ...
            
        if self.model.tick_id != self.client.tick_id:
            self.model.tick_id = self.client.tick_id
        
        if self.model.dt_step != self.client.dt_step:
            self.model.dt_step = self.client.dt_step
        
        if self.model.N != self.client.N:
            self.model.N = self.client.N
        
        self.model.accum = self.client.accum
        self.model.curr_tick_at = self.client.curr_tick_at
        self.model.next_tick_at = self.client.next_tick_at
    
        self.model.valid_indices = self.client.get_valid_indices()
        self.model.notify('valid-indices')
        
        if self.model.buffer_id != self.client.buffer_id:
            self.model.buffer_id = self.client.buffer_id
            if self.client.buffer_id == 'A':
                self.model.position = self.client.position_A
            elif self.client.buffer_id == 'B':
                self.model.position = self.client.position_B
            else:
                raise Exception("Invalid buffer ID")
        
        # Vectors
        self.model.id_to_index = self.client.id_to_index
        self.model.body_id = self.client.body_id
        self.model.flags = self.client.flags
        self.model.velocity = self.client.velocity
        self.model.mass = self.client.mass
        self.model.radius = self.client.radius
        self.model.force = self.client.force
        self.model.ledger = self.client.ledger
        self.model.cgroup = self.client.cgroup
        self.model.n_direct_contacts = self.client.n_direct_contacts
        self.model.direct_contacts = self.client.direct_contacts
        self.model.kinetic_energy = self.client.kinetic_energy
        self.model.grav_potential_energy = self.client.grav_potential_energy
        return True
    
    def on_clock_speed_changed(self, model, param):
        if not self.model.is_initialized: return
        self.client.set_clock_speed(self.model.clock_speed)
