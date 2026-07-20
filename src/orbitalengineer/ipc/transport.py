import json
import socket
import struct
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from typing import Any, Literal, Self, Sequence

from orbitalengineer.engine.clock import SimClock

class MessageType(IntEnum):
    ERROR = 0
    SUCCESS = 1
    INIT = 10
    CLOCK_START = 11
    CLOCK_PAUSE = 12
    CLOCK_SET_SPEED = 13
    END = 14
    STATUS = 15
    SYNC = 16
    BODY_SHIFT = 17

@dataclass
class ErrorResponse:
    error_message:str

@dataclass
class ParticleInit:
    flags: int
    position: tuple[float,float]
    velocity: tuple[float,float]
    mass: float
    radius: float

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(
            flags=input['flags'],
            position=input['position'],
            velocity=input['velocity'],
            mass=input['mass'],
            radius=input['radius']
        )

@dataclass
class InitRequest:
    particles: list
    device_id: int
    platform_id: int
    
    @classmethod
    def from_dict(cls, input:dict) -> Self: 
        return cls(
            device_id=input["device_id"],
            platform_id=input["platform_id"],
            particles=[
                ParticleInit.from_dict(p)
                for p in input["particles"]
            ]
        )

@dataclass
class ClockSetSpeedRequest:
    speed: float

@dataclass
class ShiftVectorsRequest:
    vector_name: str
    ids: Sequence[int]
    op: Literal["mul", "add"]
    offset: tuple[float, float]

@dataclass
class SharedMemoryInfo:
    name:str
    dtype:str
    size:int
    shape:list[int]

@dataclass
class SharedMemoryResponse:
    flags: SharedMemoryInfo
    position: SharedMemoryInfo
    velocity: SharedMemoryInfo
    mass: SharedMemoryInfo
    radius: SharedMemoryInfo
    force: SharedMemoryInfo

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        d = dict([
            (f.name, SharedMemoryInfo(**input[f.name]))
            for f in fields(cls)
        ])
        return cls(**d)

@dataclass
class ConfigResponse:
    G: float
    N: int
    coef_of_restitution: float
    dt_base: float
    EPS_DIST: float
    EPS_TIME: float

@dataclass
class InitResponse:
    initialized:bool
    config: ConfigResponse
    memory: SharedMemoryResponse

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(
            initialized=input['initialized'],
            config=ConfigResponse(**input['config']),
            memory=SharedMemoryResponse.from_dict(input['memory'])
        )

@dataclass
class StatusResponse:
    initialized: bool
    tick_id: int
    accum: float
    clock: SimClock

    @classmethod
    def from_dict(cls, input:dict) -> Self:
        return cls(
            initialized=input['initialized'],
            tick_id=input['tick_id'],
            accum=input['accum'],
            clock=SimClock(**input['clock']),
        )

# [protocol-version][message-type][payload-length]
HEADER = struct.Struct("!HHI")
PROTOCOL_VERSION = 1

def send_message(sock: socket.socket, message_enum: MessageType, message:Any|None=None) -> None:
    if message is None:
        header = HEADER.pack(PROTOCOL_VERSION, int(message_enum), 0)
        sock.sendall(header)
        return
    payload = json.dumps(asdict(message)).encode()
    header = HEADER.pack(PROTOCOL_VERSION, int(message_enum), len(payload))
    sock.sendall(header + payload)

def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)

def recv_message(sock: socket.socket, message_enum:type[MessageType]) -> tuple[int, IntEnum, dict]:
    header_data = recv_exact(sock, HEADER.size)
    version, raw_type, payload_length = HEADER.unpack(header_data)
    if payload_length == 0:
        return version, message_enum(raw_type), {}
    payload_data = recv_exact(sock, payload_length)
    try:
        payload = json.loads(payload_data)
    except json.decoder.JSONDecodeError as e:
        print(f"{payload_data=!r}")
        raise e
    return version, message_enum(raw_type), payload
