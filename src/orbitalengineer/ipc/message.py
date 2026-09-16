from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Literal, Self, Sequence, Protocol
from orbitalengineer.engine.orbitalcl.device import GPUStatus
from orbitalengineer.engine.orbitalcl.sim_config import SimConfig
from orbitalengineer.ipc.clock import SimClock

class SupportsFromDict(Protocol):
    @classmethod
    def from_dict(cls, d:dict) -> Self:...

@dataclass
class ErrorResponse:
    success: bool
    error_message:str
    
    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)

@dataclass
class ParticleInit:
    flags: int
    position: tuple[float,float]
    velocity: tuple[float,float]
    mass: float
    radius: float
    
    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)

@dataclass
class Device:
    platform_id: int
    device_id: int
    
    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)

@dataclass
class InitRequest:
    device: Device
    particles: list
    
    @classmethod
    def from_dict(cls, d:dict) -> Self: 
        return cls(
            device=Device.from_dict(d["device"]),
            particles=[
                ParticleInit.from_dict(p)
                for p in d["particles"]
            ]
        )   

@dataclass
class ClockUpdateRequest:
    speed: float|None = None
    running: bool|None = None

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)

@dataclass
class ShiftVectorsRequest:
    vector_name: str
    ids: Sequence[int]
    op: Literal["mul", "add"]
    offset: tuple[float, float]

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)


@dataclass
class SharedMemoryInfo:
    name:str
    dtype:str
    size:int
    shape:list[int]|tuple[int]


@dataclass
class SharedMemoryResponse:
    flags: SharedMemoryInfo
    position: SharedMemoryInfo
    velocity: SharedMemoryInfo
    mass: SharedMemoryInfo
    radius: SharedMemoryInfo
    force: SharedMemoryInfo
    #cgroup: SharedMemoryInfo

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        d = dict([
            (f.name, SharedMemoryInfo(**d[f.name]))
            for f in fields(cls)
        ])
        return cls(**d)

@dataclass
class InitResponse:
    initialized:bool
    config: SimConfig
    memory: SharedMemoryResponse

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(
            initialized=d['initialized'],
            config=SimConfig(**d['config']),
            memory=SharedMemoryResponse.from_dict(d['memory'])
        )


@dataclass
class StatusResponse:
    initialized: bool
    N: int
    tick_id: int
    accum: float
    clock: SimClock
    max_speed: float|None
    curr_tick_at: float
    next_tick_at: float
    dt_step: float
    gpu_status: GPUStatus|None

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(
            initialized=d['initialized'],
            tick_id=d['tick_id'],
            N=d['N'],
            accum=d['accum'],
            clock=SimClock(**d['clock']),
            max_speed=d['max_speed'],
            curr_tick_at=d['curr_tick_at'],
            next_tick_at=d['next_tick_at'],
            dt_step=d['dt_step'],
            gpu_status=GPUStatus(**d['gpu_status']) if d['gpu_status'] is not None else None
        )

@dataclass
class StateResponse:
    status: StatusResponse
    config: SimConfig|None
    memory: SharedMemoryResponse|None
    
    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(
            status=StatusResponse.from_dict(d['status']),
            config=SimConfig.from_dict(d['config']) if d['config'] is not None else None,
            memory=SharedMemoryResponse.from_dict(d['memory']) if d['memory'] is not None else None
        )

class MessageType(IntEnum):
    SUCCESS = 0
    ERROR = 1
    INIT_REQ = 10
    INIT_RESP = 11
    CLOCK_UPDATE = 20
    END_REQ = 13
    STATUS_REQ = 14
    STATUS_RESP = 15
    SYNC_REQ = 16
    SHIFT_VECTOR_REQ = 17
    STATE_REQ = 18
    STATE_RESP = 19
    DISCONNECT = 20
    TICK_ONCE = 21
    SUBSTEP_ONCE = 22

mtype_to_cls:dict[MessageType, SupportsFromDict|None] = {
    MessageType.SUCCESS: None,
    MessageType.ERROR: ErrorResponse,
    MessageType.INIT_REQ: InitRequest,
    MessageType.INIT_RESP: InitResponse,
    MessageType.CLOCK_UPDATE: ClockUpdateRequest,
    MessageType.STATUS_REQ: None,
    MessageType.STATUS_RESP: StatusResponse,
    MessageType.SYNC_REQ: None,
    MessageType.END_REQ: None,
    MessageType.SHIFT_VECTOR_REQ: ShiftVectorsRequest,
    MessageType.STATE_RESP: StateResponse
}
