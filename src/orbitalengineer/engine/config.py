import numpy as np

#############################################
# Simulation Bounds
#############################################

EPS_DIST = 1.0
EPS_TIME = 1e-4

DV_MAX = 1e10

#############################################
# Simulation Settings
#############################################

COEF_OF_RESTITUTION = 0.99
GRAV_CONSTANT = 1.0
MAX_NUM_CONTACTS_PER_BODY = 8

#############################################
# Tick-rate Settings
#############################################

DEFAULT_DT_BASE =  1/20.0
DEFAULT_SPEED = 1.0
MAX_STEPS_PER_TICK = int(1.0/np.sqrt(DEFAULT_DT_BASE))
MAX_SUB_STEPS = 100
TARGET_TICKS_PER_SECOND = 10

#############################################
# Features
#############################################

COLLISION_MERGE_ENABLE = True
COLLISION_BOUNCE_ENABLE = True
NUDGE_ON_START_ENABLE = True

#############################################
# IPC Settings
#############################################

ENABLE_PROFILING = True
METRIC_SOCKET_PATH = "/tmp/kernel-metrics.sock"

#############################################
# SerDe Settings
#############################################

SERIALIZED_VALUE_PRECISION = 6

#############################################
# Kernel Settings
#############################################

CGROUP_ASSIGN_MAX_ITERATIONS = 200
USE_FAST_RELAXED_MATH = True