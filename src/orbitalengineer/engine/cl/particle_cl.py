from typing import Any, ClassVar, Sequence
from orbitalengineer.engine.simcontroller import Particle

#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#os.environ['PYOPENCL_CACHE'] = '0'

import numpy as np


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
        return self.ctl.time_to_impact[self.idx]
