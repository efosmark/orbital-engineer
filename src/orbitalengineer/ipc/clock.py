from dataclasses import dataclass, fields
import time
from typing import Self
from orbitalengineer.engine import config


@dataclass
class SimClock:
    duration:float = 0.0
    running:bool = False
    last_time_ms:float = 0.0
    speed:float = config.DEFAULT_SPEED
    
    def start(self) -> float:
        self.last_time_ms = time.monotonic()
        self.running = True
        return self.duration
    
    def stop(self) -> float:
        self.running = False
        return self.duration
    
    def reset(self):
        self.duration = 0.0
        self.running = False
        self.last_time_ms = 0.0
        self.speed = config.DEFAULT_SPEED
    
    def time(self) -> float:
        if self.running:
            self._update()
        return self.duration
    
    def _update(self):
        t = time.monotonic()
        self.duration += (t - self.last_time_ms) * self.speed
        self.last_time_ms = t

    def increment_by(self, dt:float) -> float:
        self.duration += dt
        return self.duration

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        r = cls()
        r.duration = input["duration"] 
        r.speed = input["speed"]
        r.running = input["running"]
        r.last_time_ms = input["last_time_ms"]
        return r

    def update(self, obj:SimClock):
        for f in fields(obj):
            setattr(self, f.name, getattr(obj, f.name))
