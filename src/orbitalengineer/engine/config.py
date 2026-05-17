import numpy as np
from orbitalengineer.engine.collision_strategy import CollisionStrategy

#############################################
# Simulation Bounds
#############################################

EPS_DIST = 1e-4
EPS_TIME = 1e-4

DV_MIN = 1e-10
DV_MAX = 1e10

#############################################
# Simulation Settings
#############################################

DEFAULT_COLLISION_STRATEGY = CollisionStrategy.BOUNCE
COEF_OF_RESTITUTION = 0.90
DEFAULT_G = 1.0

#############################################
# Tick-rate Settings
#############################################

DEFAULT_DT_BASE =  1/10.0
DEFAULT_SPEED = 1.0
MAX_STEPS_PER_TICK = int(1.0/np.sqrt(DEFAULT_DT_BASE))
MAX_SUB_STEPS = 5

#############################################
# IPC Settings
#############################################

EMIT_METRICS = True
METRIC_SOCKET_PATH = "/tmp/kernel-metrics.sock"
