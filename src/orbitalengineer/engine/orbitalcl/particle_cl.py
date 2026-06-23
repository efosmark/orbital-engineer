import time
from typing import Any, ClassVar, Sequence
import numpy as np

from orbitalengineer.engine import twobody
from orbitalengineer.engine.particle import Particle


class ParticleCL(Particle):
    """ Convenience class for accessing body properties.

        This should not be used for bulk operations, but rather for cases when
        dealing with one or two specific bodies.
    """
    instances:ClassVar[dict[int, Particle]] = dict()
    ctl:Any
    
    def __new__(cls, idx:int, ctl):
        if idx not in cls.instances:
            self = object.__new__(cls)
            self.ctl = ctl
            self.idx = idx
            cls.instances[idx] = self
        return cls.instances[idx]

    def get_position(self):
        return np.complex128(self.ctl.position[self.idx])
    
    def get_xy(self) -> tuple[float,float]:
        return self.ctl.position[self.idx].real, self.ctl.position[self.idx].imag

    def get_velocity(self):
        return np.complex128(self.ctl.velocity[self.idx])
    
    def get_mass(self):
        return self.ctl.mass[self.idx]

    def get_radius(self):
        return self.ctl.radius[self.idx]

    def get_status(self):
        return 0
    
    def get_all_impact_times(self) -> Sequence[float]:
        # This items ID is available via self.idx
        # self.ctl.toi is the time-of-impact ndarray. Pairwise triu-indexed.
        # The pairwise indices can be found in self.ctl.ix and self.ctl.jx
        # Since the self-impact-time is undefined, set it to negative infinity.
        impact_times = np.full(self.ctl.N, np.inf, dtype=self.ctl.toi.dtype)
        idx = self.idx

        left_mask = self.ctl.ix == idx
        if np.any(left_mask):
            impact_times[self.ctl.jx[left_mask]] = self.ctl.toi[left_mask]

        right_mask = self.ctl.jx == idx
        if np.any(right_mask):
            impact_times[self.ctl.ix[right_mask]] = self.ctl.toi[right_mask]

        impact_times[idx] = -np.inf
        assert impact_times.size == self.ctl.N
        return impact_times # type:ignore
    
    def get_min_toi(self) -> float:
        return self.ctl._interaction.node_dt[self.idx]

    def get_flags(self) -> int:
        return self.ctl.flags[self.idx]
    
    def get_focus(self) -> tuple[int,int]|None:
        idx_start = self.idx * self.ctl.N
        idx_stop = idx_start + self.ctl.N
        
        relative_distance = np.abs(self.ctl.position - self.ctl.position[self.idx])
        accel = np.abs(np.nan_to_num(self.ctl._velocity._acceleration[idx_start:idx_stop]))
        accel[self.idx] = 0.0
        accel_over_distance = np.true_divide(accel, relative_distance + 1e-15)
        accuracy = np.true_divide(accel_over_distance, np.sum(accel_over_distance) + 1e-15)

        focus_idx = np.argmax(accel_over_distance)    
        if focus_idx is None or focus_idx == self.idx:return
        return int(focus_idx), accuracy[focus_idx]

    def get_orbit_info(self, circular_eps:float=1e-12) -> twobody.TwoBody|None:
        secondary_idx = self.idx
        if secondary_idx is None: return
        
        focus = self.get_focus()
        if focus is None:return
        focus_idx, accuracy = focus
        
        #if self.ctl.mass[secondary_idx] > self.ctl.mass[focus_idx]:
        #    secondary_idx, focus_idx = focus_idx, secondary_idx
        
        # Orbital state parameters
        r1 = (self.ctl.position[focus_idx].real, self.ctl.position[focus_idx].imag)
        r2 = (self.ctl.position[secondary_idx].real, self.ctl.position[secondary_idx].imag)
        v1 = (self.ctl.velocity[focus_idx].real, self.ctl.velocity[focus_idx].imag)
        v2 = (self.ctl.velocity[secondary_idx].real, self.ctl.velocity[secondary_idx].imag)
        m1 = self.ctl.mass[focus_idx]
        m2 = self.ctl.mass[secondary_idx]
        
        o = twobody.TwoBody(r1, r2, v1, v2, m1, m2)
        o.accuracy = accuracy
        if not o.is_bound:return
        return o
    