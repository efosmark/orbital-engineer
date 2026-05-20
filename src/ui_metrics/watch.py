import os
import socket
import json
from collections import defaultdict
from orbitalengineer.engine.config import METRIC_SOCKET_PATH


if __name__ == "__main__":

    try:
        os.unlink(METRIC_SOCKET_PATH)
    except FileNotFoundError:
        pass 

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(METRIC_SOCKET_PATH)
    
    max_metrics_per_poll = 10

    while True:
        try:
            data, _addr = sock.recvfrom(65536)
        except BlockingIOError:
            break

        try:
            metric = json.loads(data)
        except json.JSONDecodeError:
            print("bad metric:", data)
            continue

        tick_id = metric.get("value")
        if tick_id is None:
            continue

        print()
        print(f"===[  tick={tick_id}    dt={metric.get('dt_step', 0):.6f}  ]===============")
        
        
        timeline = metric.get("timeline", [])
        
        duration_by_name = defaultdict(list)
        for m in timeline:
            duration_by_name[m["name"]].append(m["duration_ms"])
        
        for name, durations in sorted(duration_by_name.items(), key=lambda x:sum(x[1]), reverse=True):
            total = sum(durations)
            N = len(durations)
            avg = total / N
            print(f"{name[:28]:<28s} [ # {N:<2.0f} | avg {avg:<6.3f} | min {min(durations):<6.3f} | max {max(durations):<6.3f} | sum {total:<6.3f} ]")