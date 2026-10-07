from contextlib import contextmanager
from dataclasses import dataclass
import time

import pyopencl as cl

@dataclass
class TraceEvent:
    name: str
    event: cl.Event|None

    def __init__(self, name:str, event: cl.Event|None=None, start:float|None=None, end:float|None=None):
        self.name = name
        self.event = event
        self._start = start
        self._end = end
    
        if self.event is None and (self._start is None or self._end is None):
            raise Exception("")
    
    @property
    def start_ns(self) -> float:
        if self.event:
            self.event.wait()
            return self.event.profile.start
        if self._start is None:
            raise ValueError("TraceEvent needs either a cl.Event or a set of start/end times")
        return self._start
    
    @property
    def end_ns(self) -> float:
        if self.event:
            self.event.wait()
            return self.event.profile.end
        if self._end is None:
            raise ValueError("TraceEvent needs either a cl.Event or a set of start/end times")
        return self._end

    @property
    def duration_ns(self) -> float:
        return self.end_ns - self.start_ns

class EventTracer:
    def __init__(self, ctl):
        self.ctl = ctl
        self.clear()
    
    def clear(self):
        self.records: list[TraceEvent] = []

    def add(self, name:str, event: cl.Event|None=None, start:float|None=None, end:float|None=None) -> cl.Event:
        if self.ctl.cfg.ENABLE_PROFILING: # type:ignore
            self.records.append(TraceEvent(name=name, event=event, start=start, end=end))
        return event #type:ignore

    def timeline(self):
        return self.records
    
    @contextmanager
    def __call__(self, name):
        start_ns = time.monotonic_ns()
        yield
        self.add(name, start=start_ns, end=time.monotonic_ns())
