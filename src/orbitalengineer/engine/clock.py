import time

class SimClock:
    
    def __init__(self, auto_start=True):
        self._duration = 0.0
        self.running = False
        if auto_start:
            self.start()
    
    def start(self) -> float:
        self._last_time_ms = time.monotonic()
        self.running = True
        return self._duration
    
    def stop(self) -> float:
        self.running = False
        return self._duration
    
    def time(self) -> float:
        if self.running:
            self._update()
        return self._duration
    
    def real_time(self) -> float:
        return time.monotonic()
    
    def _update(self):
        t = time.monotonic()
        self._duration += (t - self._last_time_ms)
        self._last_time_ms = t

    def increment_by(self, dt:float) -> float:
        self._duration += dt
        return self._duration


if __name__ == "__main__":
    import time
    
    c = SimClock()
    print(c.time())
    
    time.sleep(1.01)
    print(c.time())
    