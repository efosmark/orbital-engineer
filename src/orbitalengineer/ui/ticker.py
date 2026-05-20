import atexit
import threading
import time

from orbitalengineer.engine.simcontroller import OrbitalSimController
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.ui.gtk4 import GLib
from orbitalengineer.ui.model import DataModel

class TickController:
        
    def __init__(self, data:DataModel, orbit_ctl:OrbitalSimController, clock:SimClock):
        self.data = data
        self.orbit_ctl = orbit_ctl
        self.clock = clock
        self.logic_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.logic_running = False
        atexit.register(self.stop)
    
    def _logic_loop(self):
        next_tick_at = self.clock.time()
        while self.logic_running:
            now = self.clock.time()
            if now < next_tick_at:
                time.sleep(next_tick_at - now)
                continue

            if self.orbit_ctl.tick(now) > 0:
                GLib.idle_add(self.orbit_ctl.sync)

            now = self.clock.time()
            next_tick_at += (self.orbit_ctl.dt_base / self.orbit_ctl.speed) * 0.8
            if next_tick_at < now:
                next_tick_at = now
    
    def start(self):
        self.logic_running = True
        self.logic_thread.start()
    
    def stop(self):
        self.logic_running = False
