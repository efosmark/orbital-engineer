import cmath
import numpy as np
from dataclasses import dataclass


class PrettyReprMixin:
    _repr_precision = 3

    def __repr__(self):
        def fmt(v):
            if isinstance(v, (float, complex)):
                if not cmath.isfinite(v):
                    return repr(v)
                return f"{v:.{self._repr_precision}f}"
            if isinstance(v, np.ndarray):
                return np.array2string(
                    v,
                    precision=self._repr_precision,
                    suppress_small=True
                )
            return repr(v)

        parts = ", ".join(
            f"{k}={fmt(v)}"
            for k, v in self.__dict__.items()
        )
        return f"--{self.__class__.__name__}({parts})"



class Event:...

class CollisionEvent(Event):...

@dataclass(repr=False)
class BouncingCollisionEvent(CollisionEvent, PrettyReprMixin):
    
    tick_id:int
    i:int
    j:int
    collision_point:complex
    relative_velocity_along_normal:complex
    edge_distance:float