from dataclasses import dataclass
import math
from typing import Any, ClassVar, Sequence
import numpy as np

from orbitalengineer.engine import twobody
from orbitalengineer.engine.particle import Particle
from orbitalengineer.ui import vec
from orbitalengineer.ui.fmt import mag_format

@dataclass
class OrbitInfo:
    
    secondary_idx:int
    focus_idx:int
    
    # Orbital elements
    orbital_energy: float
    standard_grav_param: float
    argument_of_periapsis: float
    eccentricity_vec: vec.Vec2
    eccentricity: float
    orbital_period: float
    true_anomaly: float
    mean_anomaly: float
    
    # Ellipse 
    ellipse_center: vec.Vec2
    axis_a: float
    axis_b: float
    
    r_rel: vec.Vec2
    v_rel: vec.Vec2
    eccentric_anomaly: float
    v_esc: float
    direction: str
    time_periapsis: float

    def __str__(self):
        fields = [
            ('ε', f"{self.orbital_energy:.1e}"),
            #('μ', f"{self.standard_grav_param:.3e}"),
            ('θ', f"{self.true_anomaly:.2f}"),
            ('e', f"{self.eccentricity:.2f}"),
            ('ω', f"{self.argument_of_periapsis:.2f}"),
            ('M', f"{self.mean_anomaly:.1f}"),
            ('T', f"{self.orbital_period:.1f}"),
            ('τ', f"+{self.time_periapsis:.1f}"),
            ('d', f"{self.direction}"),
        ]
        return ';'.join([ f"{a}={b}" for a,b in fields ])


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
    
    def _get_focus_idx(self) -> int|None:
        idx_start = self.idx * self.ctl.N
        idx_stop = idx_start + self.ctl.N
        A = np.abs(self.ctl._velocity._acceleration)[idx_start:idx_stop]
        
        focus_idx = np.argmax(A)
        if focus_idx is None or focus_idx == self.idx:return
        
        return int(focus_idx)

    def get_orbit_info(self, circular_eps:float=1e-12) -> Any:
        secondary_idx = self.idx
        if secondary_idx is None: return
        
        focus_idx = self._get_focus_idx()
        if focus_idx is None:return
        
        if self.ctl.mass[secondary_idx] > self.ctl.mass[focus_idx]:
            secondary_idx, focus_idx = focus_idx, secondary_idx
        
        # Orbital state parameters
        r1 = (self.ctl.position[focus_idx].real, self.ctl.position[focus_idx].imag)
        r2 = (self.ctl.position[secondary_idx].real, self.ctl.position[secondary_idx].imag)
        v1 = (self.ctl.velocity[focus_idx].real, self.ctl.velocity[focus_idx].imag)
        v2 = (self.ctl.velocity[secondary_idx].real, self.ctl.velocity[secondary_idx].imag)
        m1 = self.ctl.mass[focus_idx]
        m2 = self.ctl.mass[secondary_idx]
        
        r_rel, v_rel = twobody.relative_state(r1, v1, r2, v2)
        standard_grav_param = twobody.standard_grav_param(m1, m2, 1.0)
        eccentricity_vec = twobody.ecc_vector(r_rel, v_rel, standard_grav_param)
        eccentricity = twobody.ecc_scalar(eccentricity_vec)
        a, b = twobody.ellipse_axes(r_rel, v_rel, standard_grav_param, eccentricity)

        if eccentricity < circular_eps:
            b = a
            return

        cx, cy = twobody.ellipse_center(r1, eccentricity_vec, eccentricity, a)

        e_unit = vec.unit(eccentricity_vec)
        if e_unit is None: raise ValueError(f"{e_unit=}")
            
        argument_of_periapsis = twobody.argument_of_periapsis(e_unit)

        e_unit = vec.unit(eccentricity_vec)
        if e_unit is None: return
        
        true_anomaly = (math.atan2(r_rel[1], r_rel[0]) - argument_of_periapsis)
        if true_anomaly < 0:
            true_anomaly += math.pi * 2.0
        true_anomaly = true_anomaly % (math.pi * 2.0)


        #true_anomaly = twobody.true_anomaly(eccentricity_vec, r_rel)
        #if true_anomaly is None: return

        orbital_energy = twobody.orbital_energy(standard_grav_param, vec.norm(r_rel), vec.norm(v_rel))
        eccentric_anomaly = twobody.eccentric_anomaly(true_anomaly, eccentricity_vec)
        mean_anomaly = twobody.mean_anomaly(eccentricity_vec, eccentric_anomaly)
        orbital_period = twobody.orbital_period(a, standard_grav_param)
        v_esc = np.sqrt((2 * standard_grav_param) / vec.norm(r_rel))

        direction = "prograde"
        time_periapsis = (orbital_period * (mean_anomaly / (np.pi*2.0)))
        if twobody.is_retrograde(r_rel, v_rel):
            direction = "retrograde"
            time_periapsis = orbital_period - time_periapsis
        
        return OrbitInfo(
            secondary_idx=int(secondary_idx),
            focus_idx=int(focus_idx),
            
            # Orbital elements
            orbital_energy=orbital_energy,
            standard_grav_param=standard_grav_param,
            argument_of_periapsis=argument_of_periapsis,
            eccentricity_vec=eccentricity_vec,
            eccentricity=eccentricity,
            orbital_period=orbital_period,
            true_anomaly=true_anomaly,
            mean_anomaly=mean_anomaly,
            
            # Ellipse 
            ellipse_center=(cx,cy),
            axis_a=a,
            axis_b=b,
            
            r_rel=r_rel,
            v_rel=v_rel,
            eccentric_anomaly=eccentric_anomaly,
            v_esc=v_esc,
            direction=direction,
            time_periapsis=time_periapsis
        )