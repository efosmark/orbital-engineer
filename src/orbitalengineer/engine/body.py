from dataclasses import dataclass


@dataclass
class OrbitalBody:
    x: float
    y: float
    mass: float    
    vx: float
    vy: float
    radius: float
    id:int|None=None