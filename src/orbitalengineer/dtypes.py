import numpy as np

from orbitalengineer.engine import config

COMPLEX = np.complex64
REAL = np.float32

if config.USE_DOUBLE_PRECISION:
    COMPLEX = np.complex128
    REAL = np.float64