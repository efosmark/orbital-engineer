import numpy as np
from orbitalengineer.engine.collision_strategy import CollisionStrategy


EPS_DIST = 1e-3
EPS_TIME = 1e-3

DV_MIN = 1e-10
DV_MAX = 1e10

DEFAULT_COLLISION_STRATEGY = CollisionStrategy.BOUNCE
COEF_OF_RESTITUTION = 0.90
DEFAULT_SPEED = 1.0
DEFAULT_DT_BASE =  1/20.0
DEFAULT_G = 1.0
MAX_STEPS_PER_FRAME = int(1/np.sqrt(DEFAULT_DT_BASE))
