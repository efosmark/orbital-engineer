import atexit
from collections import deque
import os
import threading
import time

from orbitalengineer.engine import config, logger
from orbitalengineer.engine.orbitalcl.orbitalcl import SimController_CL
from orbitalengineer.ipc.clock import SimClock

LOG_ENABLE_ENV_VAR = 'LOG_TICK_CTL'
NUM_TICK_DURATION_SAMPLES = 30

DT_MIN = config.EPS_TIME
DT_MAX = config.DEFAULT_DT_BASE

TARGET_TICKS_PER_SEC = 30
TARGET_TICK_RATE = (1.0/TARGET_TICKS_PER_SEC)

def clamp(value, val_min, val_max):
    return max(min(value, val_max), val_min)

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
        self.tick_duration_hot = deque(maxlen=NUM_TICK_DURATION_SAMPLES)
        self.dt_step = self.orbital.cfg.DEFAULT_DT_BASE
        self.last_tick_start_dt = 0
        self.total_dt_lag = 0.0
        self.curr_tick_at = self.clock.time()
        self.next_tick_at = self.clock.time()
        self.last_tick_end_dt = 0
    
    def avg_tick_duration(self):
        """Tick duration in _real_ time."""
        try:
            return sum(self.tick_duration)/len(self.tick_duration)
        except ZeroDivisionError:
            return self.orbital.cfg.DEFAULT_DT_BASE
    
    def avg_tick_duration_hot(self):
        """Tick duration in _real_ time."""
        try:
            return sum(self.tick_duration_hot)/len(self.tick_duration_hot)
        except ZeroDivisionError:
            return self.orbital.cfg.DEFAULT_DT_BASE
    
    def _logic_loop(self):
        while self.logic_running:
            self.dt_step = self._compute_dt_step()
            self._run_tick()

    def _compute_dt_step(self) -> float:
        dt_step = (TARGET_TICK_RATE * self.clock.speed) / 1.1
        return clamp(dt_step, DT_MIN, DT_MAX)

    def _run_tick(self):
        self.curr_tick_at = self.clock.time()
        dt_diff = self.next_tick_at - self.curr_tick_at

        if os.environ.get(LOG_ENABLE_ENV_VAR):
            avg_tick_real_dt = self.avg_tick_duration_hot()
            avg_sim = avg_tick_real_dt*self.clock.speed
            tick_rate = 1.0/TARGET_TICKS_PER_SEC
            logger.info(f"[now T{self.curr_tick_at:+3.4f}]   [next T{self.next_tick_at:+3.4f}]   [diff {dt_diff:+3.4f}]")
            logger.info(f"[dt  {self.dt_step:3.4f}]  [avg {avg_tick_real_dt:3.4f}]  [avg_sim {avg_sim:3.4f}]  [max {tick_rate/avg_tick_real_dt:3.3f}x]  [tick_rate={tick_rate:.3f}]")

        if dt_diff > 0:
            sleep_for = dt_diff / (self.clock.speed or 1.0)
            time.sleep(sleep_for)
            return
        
        tick_start = time.monotonic()
        count, dt_unprocessed = self.orbital.tick(self.dt_step)
        if count > 0:
            if os.environ.get(LOG_ENABLE_ENV_VAR):
                logger.info(f"[count {count}]  [unprocessed {dt_unprocessed:.4f}]")
            t = time.monotonic()
            self.tick_duration_hot.append(t - tick_start)
            if self.last_tick_end_dt > 0:
                self.tick_duration.append(t - self.last_tick_end_dt)
                self.last_tick_end_dt = t
            self.last_tick_end_dt = t
            self.clock.decrement_by(float(dt_unprocessed))
        
        self.total_dt_lag = -dt_diff if dt_diff < 0 else 0
        self.next_tick_at += self.dt_step

    def start(self):
        self.logic_running = True
        self.logic_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.logic_thread.start()
    
    def stop(self):
        self.logic_running = False
