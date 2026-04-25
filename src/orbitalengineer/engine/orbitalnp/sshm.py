import atexit
import math
from multiprocessing import shared_memory
from types import EllipsisType
from typing import Any, Callable, cast
from typing import get_args, get_origin


import numpy as np
from numpy.typing import NDArray

from orbitalengineer.engine import logger
from orbitalengineer.ui.fmt import mag_format

N = -1

class field(Any):
    shape:tuple
    dtype:type|None
    default: Callable[[int],Any]|Any|None
    
    def __init__(self, *shape:int|EllipsisType, dtype:type|None=None, default:Callable[[int],Any]|Any|None=None):
        self.shape = shape
        self.dtype = dtype
        self.default = default

class FieldMeta(type):
    """
    A metaclass that automatically registers subclasses.
    """
    _fields:dict[str, field]
    
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        annotations = namespace.get("__annotations__", {})
        if not hasattr(cls, 'fields'):
            cls._fields = {}
        
        for prop in dir(cls):
            fv = getattr(cls, prop)
            if isinstance(fv, field):
                cls._field_dtype(prop, fv, annotations)
                cls._fields[prop] = fv
        return cls

    def _field_dtype(cls, field, fv, annotations):
        if fv.dtype is None:
            #if field not in annotations:
            #    raise TypeError(f"Field '{field}' must either have a type annotation or dtype specified.")
            
            if get_origin(annotations.get(field, None)) == np.ndarray:
                fv.dtype = get_args(get_args(annotations[field])[1])[0]
            else:
                fv.dtype = annotations.get(field, None)


class StructuredSharedMemory(shared_memory.SharedMemory, metaclass=FieldMeta):
    N:int

    def __init__(self, N:int, name:str|None=None):
        self.N = N
        self.buffer_size = self._get_buffer_size()
        if name is not None:
            shared_memory.SharedMemory.__init__(self, name=name)
        else:
            self._create_shared_memory_block()
        self._init_fields()
        if name is None:
            for name, fv in self._fields.items():
                if not fv.default:
                    continue
                if callable(fv.default):
                    getattr(self, name).fill(fv.default(self.N))
                else:
                    getattr(self, name).fill(fv.default)

    def _create_shared_memory_block(self):
        shared_memory.SharedMemory.__init__(self, create=True, size=self.buffer_size)
        atexit.register(lambda: self.unlink())
        logger.info(f"Allocated {mag_format(self.buffer_size)}B of shared memory.")
    
    def _get_shape(self, field:field):
        return tuple([
            self.N if ax == ... else ax 
            for ax in field.shape
        ])
    
    def _init_fields(self):
        offset = 0
        for name, f in self._fields.items():
            setattr(self, name, np.ndarray(self._get_shape(f), dtype=f.dtype, buffer=self.buf, offset=offset))
            offset += self._get_size(f)

    def _get_size(self, field:field):
        """Get the full byte size of a field based on its dtype and shape."""
        return math.prod([
            np.dtype(field.dtype).itemsize * (self.N if axis_size == -1 or axis_size == ... else axis_size)
            for axis_size in field.shape
        ])
        
    def _get_buffer_size(self):
        return sum([
            self._get_size(field)
            for field in self._fields.values()
        ])
    
    def copyto(self, other: 'StructuredSharedMemory'):
        for field_name, field in self._fields.items():
            np.copyto(getattr(other, field_name), getattr(self, field_name))
