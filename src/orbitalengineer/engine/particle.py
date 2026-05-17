from typing import Any, Protocol, Sequence


class Particle(Protocol):
    idx:int|None = None

    def get_xy(self) -> tuple[float,float]:
        position = self.get_position()
        return position.real, position.imag
        
    def get_position(self) -> complex:...
    def get_velocity(self) -> complex:...
    def get_mass(self) -> float:...
    def get_radius(self) -> float:...
    def get_min_toi(self) -> float:...
    def get_all_impact_times(self) -> Sequence[float]:...
    def get_status(self) -> float:...

    def get_flags(self) -> int:...
    def has_flag(self, flag:int) -> bool:
        return (self.get_flags() & flag) == flag

    def __str__(self):
        fields:list[tuple[str, Any]] = []
        if self.idx is not None:
            fields.append(('idx', f"{self.idx}"))
        fields.extend([
            ('position', f"{self.get_position():.1f}"),
            ('velocity', f"{self.get_velocity():.1f}"),
            ('radius', f"{self.get_radius():.1f}"),
            ('mass', f"{self.get_mass():.1f}"),
        ])
        field_string = ', '.join([
            f'{k}={v}' for k,v in fields
        ])
        return f"{self.__class__.__name__}({field_string})"


class ParticleRaw(Particle):
    idx:int|None = None
    
    _position:complex
    _velocity:complex
    _mass:float
    _radius:float
    _status:int
    
    def __init__(self, *, position, velocity, radius, mass, flags=0):
        self._position = position
        self._velocity = velocity
        self._radius = radius
        self._mass = mass
        self._flags = flags
    
    def get_position(self) -> complex:
        return self._position
    
    def get_xy(self) -> tuple[float,float]:
        position = self.get_position()
        return position.real, position.imag
    
    def get_velocity(self) -> complex:
        return self._velocity
    
    def get_mass(self) -> float:
        return self._mass
    
    def get_radius(self) -> float:
        return self._radius
        
    def get_status(self) -> float:
        return self._status

    def get_min_toi(self) -> float:
        return 0.0
    
    def get_all_impact_times(self) -> Sequence[float]:
        raise NotImplementedError()
    
    def get_flags(self) -> int:
        return self._flags