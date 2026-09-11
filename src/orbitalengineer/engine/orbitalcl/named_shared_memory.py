from multiprocessing import shared_memory
from pathlib import Path
from numpy.typing import NDArray
import numpy as np

from orbitalengineer.engine import logger

class NamedSharedMemory:
    shm:dict[str, shared_memory.SharedMemory]
    vec:dict[str, NDArray]
    
    def __init__(self):
        self.shm = dict()
        self.vec = dict()

    def __getitem__(self, key) -> shared_memory.SharedMemory:
        return self.shm[key]

    def create_shared_memory(self, field_name:str, size:int, dtype:type) -> NDArray:
        t = np.dtype(dtype)
        logger.info("shm: %s size=%s dtype=%s", field_name, size, t)
        self.shm[field_name] = shared_memory.SharedMemory(create=True, size=t.itemsize * size)
        self.vec[field_name]= np.ndarray(size, dtype=dtype, buffer=self.shm[field_name].buf)
        return self.vec[field_name]

    def disconnect(self):
        if not hasattr(self, 'shm'):
            return
        closed = []
        for name, shm in self.shm.items():
            try:
                shm.close()
                shm.unlink()
                closed.append(name)
            except FileNotFoundError:
                logger.warning("Could not properly close shared memory: file(s) not found.")
                ...
        logger.info("Closed shared memory: %s", ','.join(closed))