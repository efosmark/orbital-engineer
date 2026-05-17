import socket
import json
import time

class MetricsProducer:
    
    def __init__(self, sock_path:str, blocking:bool=False):
        self.socket_path = sock_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(blocking)
        
        self.sent = 0
        self.dropped = 0

    def emit_metric(self, name: str, value: float, **tags):
        metric = {
            "t_ns": time.monotonic_ns(),
            "name": name,
            "value": value,
            **tags,
        }

        data = json.dumps(metric, separators=(",", ":")).encode("utf-8")
        try:
            self.sock.sendto(data, self.socket_path)
            self.sent += 1
        except BlockingIOError as e:
            self.dropped += 1
        except FileNotFoundError:
            self.dropped += 1
        except OSError as e:
            self.dropped += 1
