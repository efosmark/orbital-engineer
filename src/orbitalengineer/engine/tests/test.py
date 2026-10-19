import atexit
from dataclasses import dataclass

from orbitalengineer.engine.controller import SimController
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.helpers import r_from_mass, vis_viva
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.message import ParticleInit
from orbitalengineer.ipc.ticker import TickController
from orbitalengineer.twobody import twobody

_Device = tuple[int,int]
DEFAULT_DEVICE:_Device = (0,0)

class OrbitalTestFixtureException(Exception):...

class BodyTest:
    """A proxy to initialize and procure body vector values."""
    def __init__(
        self,
        ctl:  'OrbitalTest',
        index:    int,
        position: complex,
        velocity: complex,
        mass:     float,
        radius:   float,
        flags:    int
    ):
        self.ctl = ctl
        self._orbital = ctl.orbital
        self.index = index
        self._position = position
        self._radius = radius
        self._velocity = velocity
        self._radius = radius
        self._mass = mass
        self._flags = flags

    @property
    def radius(self) -> float:
        if self._orbital.is_initialized:
            return self.ctl.orbital.vec.get_host_vector(self.ctl.orbital.vec.radius)[self.index]
        return self._radius

    @property
    def mass(self) -> float:
        if self._orbital.is_initialized:
            return self.ctl.orbital.vec.get_host_vector(self.ctl.orbital.vec.mass)[self.index]
        return self._mass

    @property
    def position(self) -> complex:
        if self._orbital.is_initialized:
            return self.ctl.orbital.vec.get_host_vector(self.ctl.orbital.vec.position)[self.index]
        return self._position

    @property
    def velocity(self) -> complex:
        if self._orbital.is_initialized:
            return self.ctl.orbital.vec.get_host_vector(self.ctl.orbital.vec.velocity)[self.index]
        return self._velocity

    def as_init(self):
        """The sim controller requires a slightly different format to initialize."""
        # TODO: Consolidate the implementation of ParticleInit, BodyProxy, ParticleCL, etc.
        #       Protocol class with requisite fields is probably the best option.
        return ParticleInit(
            flags=self._flags,
            position=(self._position.real, self._position.imag),
            velocity=(self._velocity.real, self._velocity.imag),
            mass=self._mass,
            radius=self._radius,
        )

def orbital_period(primary:BodyTest, secondary:BodyTest, G:float):
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
    _bodies:list[BodyTest]
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
        self.orbital.init_sim([b.as_init() for b in self._bodies])
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
            
            self.orbital.sync()
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
        b = BodyTest(self, len(self._bodies), position, velocity, mass, radius, flags)
        self._bodies.append(b)
        return b

    def add_orbiting_body(
        self,
        primary_body:BodyTest,
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
        
        b = BodyTest(
            self,
            len(self._bodies),
            position=primary_body._position + position,
            velocity=primary_body._velocity + velocity,
            mass=mass,
            radius=r_from_mass(mass) if radius is None else radius,
            flags=flags
        )
        self._bodies.append(b)
        return b
