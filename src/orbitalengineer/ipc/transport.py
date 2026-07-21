import json
import socket
import struct
from dataclasses import asdict
from enum import IntEnum
from typing import Any

from orbitalengineer.ipc.message import MessageType

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
