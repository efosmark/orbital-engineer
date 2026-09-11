import atexit
from collections import deque
import os
import threading
import time

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalcl.orbitalcl import SimController_CL
from orbitalengineer.ipc.clock import SimClock

LOG_ENABLE_ENV_VAR = 'LOG_TICK_CTL'
NUM_TICK_DURATION_SAMPLES = 20

class TickController:
        
    def __init__(self, orbital:SimController_CL, clock:SimClock):
        self.orbital = orbital
        self.clock = clock
        self.logic_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.logic_running = False
        atexit.register(self.stop)
        self.reset()
    
    def reset(self):
        self.tick_duration = deque(maxlen=NUM_TICK_DURATION_SAMPLES)
        self.dt_step = self.orbital.cfg.DEFAULT_DT_BASE
        self.accum = 0.0
        self.total_dt_lag = 0.0
        self.next_tick_at = self.clock.time()
    
    def avg_tick_duration(self):
        try:
            return sum(self.tick_duration)/len(self.tick_duration)
        except ZeroDivisionError:
            return self.orbital.cfg.DEFAULT_DT_BASE
    
    def max_target_clock_speed(self):
        avg_tick_real_dt = self.avg_tick_duration()
        return self.dt_step / avg_tick_real_dt
    
    def _logic_loop(self):
        while self.logic_running:
            self._run_tick()

    def _run_tick(self):
        now = self.clock.time()
        dt_diff = self.next_tick_at - now

        if os.environ.get(LOG_ENABLE_ENV_VAR):
            avg_tick_real_dt = self.avg_tick_duration()
            max_speed = self.max_target_clock_speed()
            logger.info(f"[now={now:.3f}]   [next_tick_at={self.next_tick_at:.3f}]   [{dt_diff=:+.3f}]   [{avg_tick_real_dt=:.3f}]   [max_spd={max_speed:.1f}]")
            logger.info(f"[accum={self.accum:.3f}]   [dt={self.dt_step:.3f}]")
                        
        if now < self.next_tick_at:
            if self.accum < self.dt_step:
                sleep_for = dt_diff / (self.clock.speed or 1.0)
                time.sleep(sleep_for)
            elif self.accum >= self.dt_step:
                # todo only try catch-up if the amount of time is less than the average tick time
                dt_unprocessed = self.orbital.tick(self.dt_step)
                dt_processed = self.dt_step - dt_unprocessed
                self.accum -= dt_processed
            return
                
        self.accum += self.dt_step
        if self.accum >= self.dt_step:
            tick_start = time.monotonic()
            dt_unprocessed = self.orbital.tick(self.dt_step)
            self.tick_duration.append(time.monotonic() - tick_start)
            dt_processed = self.dt_step - dt_unprocessed
            self.accum -= dt_processed
        
        self.total_dt_lag = self.accum + (-dt_diff if dt_diff < 0 else 0)
        self.next_tick_at += self.dt_step

    def start(self):
        self.logic_running = True
        self.logic_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.logic_thread.start()
    
    def stop(self):
        self.logic_running = False
