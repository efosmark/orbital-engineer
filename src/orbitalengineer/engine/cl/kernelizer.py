from pathlib import Path
from typing import Sequence
import pyopencl as cl
from pyopencl.typing import KernelArg

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


class KernelBuilder:
    kernel_name:str    
    kernel_src:str|None
    kernel_file:str|None
    
    def __init__(self, ctx:cl.Context, queue:cl.CommandQueue, tr:EventTracer, build_options:Sequence|None=None):
        if hasattr(self, 'kernel_src') and self.kernel_src is not None:
            kernel_src = self.kernel_src
        
        elif self.kernel_file is not None:
            kernel_src = open(Path(__file__).parent / self.kernel_file, 'r').read()
        
        else:
            raise Exception("Must specify either kernel_src or kernel_file")
        
        self.tr = tr
        
        opts = build_options or ['-cl-std=CL2.0']
        
        self.ctx = ctx
        self.queue = queue
        self.prg = cl.Program(self.ctx, kernel_src).build(
            options=[*opts]
        )
        self._kernel = cl.Kernel(self.prg, self.kernel_name)

    def kernel(self,
            queue: cl.CommandQueue,
            global_work_size: tuple[int, ...],
            local_work_size: tuple[int, ...] | None,
            *args: KernelArg,
            wait_for: cl.WaitList = None,
            g_times_l: bool = False,
            allow_empty_ndrange: bool = False,
            global_offset: tuple[int, ...] | None = None,
            metric_alias:str|None = None
         ) -> cl.Event:
        
            evt = self._kernel(
                queue,
                global_work_size,
                local_work_size,
                *args,
                wait_for=wait_for,
                g_times_l=g_times_l,
                allow_empty_ndrange=allow_empty_ndrange,
                global_offset=global_offset
            )
            
            return self.tr.add(
                metric_alias or self.kernel_name,
                evt
            )


def get_kernel_src():
    with open(Path(__file__).parent / kernel_file, 'r') as f:
        return f.read()