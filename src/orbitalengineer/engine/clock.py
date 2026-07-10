import time
from typing import Self
from orbitalengineer.engine import config

FIELDS = ["duration", "speed", "running", "last_time_ms"]

class SimClock:
    
    def __init__(self, auto_start=True):
        self.duration = 0.0
        self.running = False
        self._speed = config.DEFAULT_SPEED
        if auto_start: self.start()
    
    def start(self) -> float:
        self.last_time_ms = time.monotonic()
        self.running = True
        return self.duration
    
    def stop(self) -> float:
        self.running = False
        return self.duration
    
    def time(self) -> float:
        if self.running:
            self._update()
        return self.duration
    
    def real_time(self) -> float:
        return time.monotonic()
    
    def _update(self):
        t = time.monotonic()
        self.duration += (t - self.last_time_ms) * self._speed
        self.last_time_ms = t

    def increment_by(self, dt:float) -> float:
        self.duration += dt
        return self.duration

    @property
    def speed(self) -> float:
        if 0 <= self._speed < 1:
            return 1/self._speed
        return self._speed

    @speed.setter
    def speed(self, value):
        if value < 0:
            value = 1/float(value)
        self._speed = value

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "speed": self.speed,
            "running": self.running,
            "last_time_ms": self.last_time_ms
        }

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        r = cls()
        r.duration = input["duration"]
        r.speed = input["speed"]
        r.running = input["running"]
        r.last_time_ms = input["last_time_ms"]
        return r

    def update(self, obj):
        for field in FIELDS:
            if field not in obj:
                continue
            setattr(self, field, obj[field])
