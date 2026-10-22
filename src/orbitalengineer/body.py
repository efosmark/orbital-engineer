from typing import Protocol, Self, overload, override

import copy
copy.deepcopy
import pickle
import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine.primary_vectors import HasStateVectors

class Body:
    """A proxy to initialize and procure body vector values."""

    _position: complex
    _velocity: complex
    _mass:     float
    _radius:   float
    _flags:    int

    @overload
    def __init__(self, *, vec: HasStateVectors, index:int):...
    @overload
    def __init__(self, *, position:complex, velocity:complex, mass:float, radius:float, flags:int):...
    @overload
    def __init__(self, *, position:complex, velocity:complex, mass:float, radius:float, flags:int, vec:HasStateVectors, index:int):...
    def __init__(
        self,
        *,
        position: complex|None=None,
        velocity: complex|None=None,
        mass:     float|None=None,
        radius:   float|None=None,
        flags:    int|None=None,
        vec:      HasStateVectors|None=None,
        index:    int|None=None,
    ):
        if vec is None:
            if position is not None: self._position = position
            if radius is not None:   self._radius = radius
            if mass is not None:     self._mass = mass
            if velocity is not None: self._velocity = velocity
            if radius is not None:   self._radius = radius
            if flags is not None:    self._flags = flags
            self._vec = None
        else:
            self._vec = vec
            self.index = index

    @property
    def radius(self) -> float:
        if self._vec is not None:
            self._radius = float(self._vec.radius[self.index])
        return self._radius

    @property
    def mass(self) -> float:
        if self._vec is not None:
            self._mass = float(self._vec.mass[self.index])
        return self._mass

    @property
    def position(self) -> complex:
        if self._vec is not None:
            self._position = complex(self._vec.position[self.index])
        return self._position

    @property
    def velocity(self) -> complex:
        if self._vec is not None:
            self._velocity = complex(self._vec.velocity[self.index])
        return self._velocity

    @property
    def flags(self) -> int:
        if self._vec is not None:
            self._flags = int(self._vec.flags[self.index])
        return self._flags

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(
            position=complex(*d['position']),
            velocity=complex(*d['velocity']),
            mass=float(d['mass']),
            radius=float(d['radius']),
            flags=int(d['flags']),
        )

    def __deepcopy__(self, memo) -> dict:
        return {
            'flags': self.flags,
            'position': [round(self.position.real, 6), round(self.position.imag, 6)],
            'velocity': [round(self.velocity.real, 6), round(self.velocity.imag, 6)],
            'mass': self.mass,
            'radius': self.radius,
        }