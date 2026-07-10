from dataclasses import dataclass, asdict
from typing import Self, cast

from orbitalengineer.engine.clock import SimClock
from orbitalengineer.ipc.typed import SupportsFromDict, SupportsToDict


@dataclass
class DeviceResponse:
    device_id: int
    platform_id: int
    
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(**input)


@dataclass
class ConfigResponse:
    G: float
    N: int
    coef_of_restitution: float
    dt_base: float
    EPS_DIST: float
    EPS_TIME: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(**input)


@dataclass
class StatusResponse:
    tick_id: int
    accum: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(**input)


@dataclass
class _SharedMemoryState:
    name:str
    dtype:str
    size:int
    shape:list[int]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(**input)


class SharedMemoryStateResponse:
    _FIELDS = ['flags', 'position', 'velocity', 'mass', 'radius', 'force']

    flags:_SharedMemoryState
    position:_SharedMemoryState
    velocity:_SharedMemoryState
    mass:_SharedMemoryState
    radius:_SharedMemoryState
    force:_SharedMemoryState

    def to_dict(self) -> dict:
        d = {}
        for field in self._FIELDS:
            v = cast(SupportsToDict, getattr(self, field))
            if v is None: continue
            d[field] = v.to_dict()
        return d

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        r = cls()
        for field in cls._FIELDS:
            setattr(r, field, _SharedMemoryState.from_dict(input[field]))        
        return r

class ServerResponse:
    _OP_CONFIRM_FIELDS = ['load', 'init', 'sync', 'start', 'stop', 'reset', 'tick']
    _INFO_FIELDS:dict[str, SupportsFromDict] = {
        'device': DeviceResponse,
        'config': ConfigResponse,
        'clock': SimClock,
        'memory': SharedMemoryStateResponse,
        'status': StatusResponse
    }
    
    # Operation confirmations
    load:bool|None = None
    init:bool|None = None
    sync:bool|None = None
    start:bool|None = None
    stop:bool|None = None
    reset:bool|None = None
    tick:bool|None = None
    
    # Information
    device:DeviceResponse|None = None
    config:ConfigResponse|None = None
    clock:SimClock|None = None
    memory:SharedMemoryStateResponse|None = None
    status:StatusResponse|None = None
    
    def to_dict(self) -> dict:
        resp = {}
        for field in self._OP_CONFIRM_FIELDS:
            v = getattr(self, field)
            if v is not None:
                resp[field] = v
        for field in self._INFO_FIELDS:
            v = cast(SupportsToDict, getattr(self, field))
            if v is not None:
                resp[field] = v.to_dict()
        return resp
    
    @classmethod
    def from_dict(cls, input:dict) -> Self:
        resp = cls()
        for field in cls._OP_CONFIRM_FIELDS:
            if field not in input or input[field] is None: continue
            setattr(resp, field, input[field])
        for field, container in cls._INFO_FIELDS.items():
            if field not in input or input[field] is None: continue
            setattr(resp, field, container.from_dict(input[field]))
        return resp