import atexit
from dataclasses import dataclass

from orbitalengineer.engine.controller import SimController
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.helpers import r_from_mass, vis_viva
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.ticker import TickController
from orbitalengineer.twobody import twobody
from orbitalengineer import body

_Device = tuple[int,int]
DEFAULT_DEVICE:_Device = (0,0)

class OrbitalTestFixtureException(Exception):...


def orbital_period(primary:body.Body, secondary:body.Body, G:float):
    a = abs(secondary._position - primary._position)
    mu = twobody.standard_grav_param(primary.mass, secondary.mass, G=G)
    return twobody.orbital_period(a, mu)

@dataclass
class TickResult:
    tick_id:int
    num_steps:int
    dt_remaining:float
    dt_processed:float
    position:tuple[complex]
    velocity:tuple[complex]
    mass:tuple[float]
    radius:tuple[float]
    flags:tuple[int]


class OrbitalTest:
    config:SimConfig
    _bodies:list[body.Body]
    _configured:bool = False
    _initialized:bool = False
    _executed:bool = False

    def __init__(self, device:_Device=DEFAULT_DEVICE):
        self.orbital = SimController()
        self.orbital.set_cl_device(*device)
        self.clock = SimClock()
        self.tick_ctl = TickController(self.orbital, self.clock)
        self.reset()
        
        atexit.register(self.orbital.shm.disconnect)
    
    def reset(self):
        self.orbital.reset()
        self.config = SimConfig()
        self._initialized = False
        self._executed = False
        self._bodies = []
    
    def _initialize(self):
        self.orbital.init_sim([b for b in self._bodies])
        self._initialized = True

    def run_for(self, run_time:float, dt_step:float|None=None):
        if not self._initialized:
            self._initialize()
        if dt_step is None:
            dt_step = self.config.DEFAULT_DT_BASE
        
        self.orbital.cfg = self.config
        
        dt_remaining = run_time
        
        tick_states = []
        num_ticks = 0
        num_sub_steps = 0
        while dt_remaining >= self.config.EPS_TIME:
            dt = dt_remaining if dt_remaining < dt_step else dt_step
            dt_remaining -= dt
            c, unp = self.orbital.tick(dt)
            dt_remaining += unp
            num_sub_steps += c
            num_ticks += 1
            
            self.orbital.sync('A')
            tick_states.append(TickResult(
                num_ticks,
                num_sub_steps,
                dt_remaining=dt_remaining,
                dt_processed=dt - unp,
                position=self.orbital.vec.get_host_vector(self.orbital.vec.position).tolist(),
                velocity=self.orbital.vec.get_host_vector(self.orbital.vec.velocity).tolist(),
                mass=self.orbital.vec.get_host_vector(self.orbital.vec.mass).tolist(),
                radius=self.orbital.vec.get_host_vector(self.orbital.vec.radius).tolist(),
                flags=self.orbital.vec.get_host_vector(self.orbital.vec.flags).tolist(),
            ))
            if c > 0:
                self.clock.decrement_by(float(unp))
        return tick_states

    def add_body(self, *, position, velocity, radius, mass, flags=0):
        if self._executed:
            raise OrbitalTestFixtureException("Cannot add_body after the fixture has been executed.")
        b = body.Body(
            position=position,
            velocity=velocity,
            mass=mass,
            radius=radius,
            flags=flags,
            vec=self.orbital.vec.host,
            index=len(self._bodies)
        )
        self._bodies.append(b)
        return b

    def add_orbiting_body(
        self,
        primary_body:body.Body,
        *,
        position:complex,
        mass:float=0,
        radius:float|None=None,
        flags:int=0,
        prograde:bool=True,
        eccentricity:float=1.0,
    ):

        dist = abs(position)
        velocity = vis_viva(
            position=position,
            mu=(self.config.GRAV_CONSTANT * (primary_body._mass + mass)),
            a=dist * eccentricity,
            prograde=prograde
        )
        
        b = body.Body(
            position=primary_body._position + position,
            velocity=primary_body._velocity + velocity,
            mass=mass,
            radius=r_from_mass(mass) if radius is None else radius,
            flags=flags,
            vec=self.orbital.vec.host,
            index=len(self._bodies),
        )
        self._bodies.append(b)
        return b
