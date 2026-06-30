import numpy as np

#############################################
# Simulation Bounds
#############################################

EPS_DIST = 1e-3
EPS_TIME = 1e-3

DV_MIN = 1e-10
DV_MAX = 1e10

#############################################
# Simulation Settings
#############################################

COEF_OF_RESTITUTION = 0.90
DEFAULT_G = 1.0

#############################################
# Tick-rate Settings
#############################################

DEFAULT_DT_BASE =  1/20.0
DEFAULT_SPEED = 1.0
MAX_STEPS_PER_TICK = int(1.0/np.sqrt(DEFAULT_DT_BASE))
MAX_SUB_STEPS = 5

#############################################
# Features
#############################################

COLLISION_MERGE_ENABLE = True
COLLISION_BOUNCE_ENABLE = True

#############################################
# IPC Settings
#############################################

EMIT_METRICS = True
METRIC_SOCKET_PATH = "/tmp/kernel-metrics.sock"

#############################################
# SerDe Settings
#############################################

SERIALIZED_VALUE_PRECISION = 6