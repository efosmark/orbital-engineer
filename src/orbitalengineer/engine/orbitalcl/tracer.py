from contextlib import contextmanager
import time

import pyopencl as cl

class TraceEvent:
    name: str
    event: cl.Event|None

    def __init__(self, name:str, event: cl.Event|None=None, start:int|None=None, end:int|None=None):
        self.name = name
        
        self.event = event
        self._start = start
        self._end = end
    
        if self.event is None and (self._start is None or self._end is None):
            raise Exception("")
    
    @property
    def start_ns(self) -> int:
        if self.event:
            self.event.wait()
            return self.event.profile.start
        if self._start is None:
            raise ValueError("TraceEvent needs either a cl.Event or a set of start/end times")
        return self._start
    
    @property
    def end_ns(self) -> int:
        if self.event:
            self.event.wait()
            return self.event.profile.end
        if self._end is None:
            raise ValueError("TraceEvent needs either a cl.Event or a set of start/end times")
        return self._end

class EventTracer:
    def __init__(self, ctl):
        self.ctl = ctl
        self.clear()
    
    def clear(self):
        self.records: list[TraceEvent] = []

    def add(self, name:str, event: cl.Event|None=None, start:int|None=None, end:int|None=None) -> cl.Event:
        if self.ctl.cfg.ENABLE_PROFILING: # type:ignore
            self.records.append(TraceEvent(name=name, event=event, start=start, end=end))
        return event #type:ignore

    def timeline(self):
        # Assumes queue profiling is enabled and all commands are complete
        rows = []
        if not self.records:
            return rows

        starts = [r.start_ns for r in self.records]
        origin = min(starts)

        for r in self.records:
            rows.append({
                "name": r.name,
                "start_ns": r.start_ns - origin,
                "end_ns": r.end_ns - origin,
                "duration_ns": r.end_ns - r.start_ns,
            })

        rows.sort(key=lambda row: row["start_ns"])
        return rows
    
    @contextmanager
    def __call__(self, name):
        start_ns = time.monotonic_ns()
        yield
        self.add(name, start=start_ns, end=time.monotonic_ns())
