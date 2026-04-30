import pyopencl as cl

from orbitalengineer.engine.simcontroller import OrbitalSimController
mf = cl.mem_flags

import pyopencl as cl
from dataclasses import dataclass

@dataclass
class TraceEvent:
    name: str
    event: cl.Event

class EventTracer:
    def __init__(self, ctl: OrbitalSimController):
        self.ctl = ctl
        self.clear()
    
    def clear(self):
        self.records: list[TraceEvent] = []

    def add(self, name: str, event: cl.Event) -> cl.Event:
        if self.ctl.enable_profiling: # type:ignore
            self.records.append(TraceEvent(name=name, event=event))
        return event

    def timeline(self):
        # Assumes queue profiling is enabled and all commands are complete
        rows = []
        if not self.records:
            return rows

        starts = [r.event.profile.start for r in self.records]
        origin = min(starts)

        for r in self.records:
            p = r.event.profile
            rows.append({
                "name": r.name,
                "start_ns": p.start - origin,
                "end_ns": p.end - origin,
                "duration_ns": p.end - p.start,
            })

        rows.sort(key=lambda row: row["start_ns"])
        return rows