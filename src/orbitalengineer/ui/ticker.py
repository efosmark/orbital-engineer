import atexit
import threading
import time

from orbitalengineer.engine.cl.orbitalcl import TickStats
from orbitalengineer.ui.gtk4 import GLib
from orbitalengineer.ui.model import DataModel
from orbitalengineer.engine.simcontroller import OrbitalSimController

class TickController:
        
    def __init__(self, data:DataModel, orbit_ctl:OrbitalSimController):
        self.data = data
        self.orbit_ctl = orbit_ctl
        self.logic_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.logic_running = False
        atexit.register(self.stop)
    
    def _logic_loop(self):
        next_tick_at = time.monotonic()
        while self.logic_running:
            now = time.monotonic()
            if now < next_tick_at:
                time.sleep(next_tick_at - now)
                continue

            stats = self.orbit_ctl.tick(now)
            
            if stats.num_steps > 0:
                def _add_durations(stats: TickStats):
                    self.orbit_ctl.sync()
                    self.data.num_ticks = stats.tick_id
                    for m in stats.metrics:
                        self.data.add_duration(m.field, m.value / 1000.0, stats.tick_id)
                    return False
                GLib.idle_add(_add_durations, stats)

            now = time.monotonic()
            #next_tick_at = now
            next_tick_at += (self.orbit_ctl.dt_base / self.orbit_ctl.speed) * 0.5
            if next_tick_at < now:
                next_tick_at = now
                    
    def start(self):
        self.logic_running = True
        self.logic_thread.start()
    
    def stop(self):
        self.logic_running = False
