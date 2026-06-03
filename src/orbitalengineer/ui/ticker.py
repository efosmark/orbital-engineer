import atexit
import threading
import time

from orbitalengineer.engine.cl.orbitalcl import SimController_CL
from orbitalengineer.engine.clock import SimClock
from orbitalengineer.ui.gtk4 import GLib

class TickController:
        
    def __init__(self, orbital:SimController_CL, clock:SimClock):
        self.orbital = orbital
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

            if self.orbital.tick(now) > 0:
                GLib.idle_add(self.orbital.sync)

            now = self.clock.time()
            next_tick_at += (self.orbital.dt_base / self.clock.speed) * 0.4
            if next_tick_at < now:
                next_tick_at = now
    
    def start(self):
        self.logic_running = True
        self.logic_thread.start()
    
    def stop(self):
        self.logic_running = False
