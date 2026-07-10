import json
import socket

from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.orbitalcl import orbitalcl
from orbitalengineer.ipc.response import _SharedMemoryState, ConfigResponse, DeviceResponse, ServerResponse, SharedMemoryStateResponse, StatusResponse
from orbitalengineer.ipc.ticker import TickController

HOST = ''        # Symbolic name meaning all available interfaces
PORT = 50008     # Arbitrary non-privileged port

class OrbitalControlServer:
    tick_ctl:TickController

    def __init__(self):
        self.orbital = orbitalcl.SimController_CL()
        self.clock = SimClock()

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"Orbital Server started. {HOST=} {PORT=}")
            while True:
                conn, addr = s.accept()
                self._handle_client(conn, addr)
        self.orbital.disconnect()

    def _handle_client(self, conn, addr):
        print("Connection from", addr)
        try:
            buffer = b''
            while True:
                data = conn.recv(1024)
                buffer += data
                if not data: break
                if b'\n' not in buffer: continue
                
                data, buffer = buffer.split(b'\n', 1)
                parsed = json.loads(data)
                response = self.handle_request(parsed)
                conn.sendall(json.dumps(response.to_dict()).encode() + b"\n")
        except KeyboardInterrupt:
            self.orbital.disconnect()
        
        if hasattr(self, 'tick_ctl'):
            self.tick_ctl.stop()

    def _get_shared_memory_state(self) -> SharedMemoryStateResponse:
        def _get_shared_memory_state(field):
            return _SharedMemoryState(
                name=self.orbital.shm[field].name,
                dtype=str(getattr(self.orbital, field).dtype),
                size=self.orbital.shm[field].size,
                shape=getattr(self.orbital, field).shape
            )  
        m = SharedMemoryStateResponse()
        m.flags = _get_shared_memory_state('flags')
        m.velocity = _get_shared_memory_state('velocity')
        m.position = _get_shared_memory_state('position')
        m.mass = _get_shared_memory_state('mass')
        m.radius = _get_shared_memory_state('radius')
        m.force = _get_shared_memory_state('force')
        return m    
    
    def handle_request(self, req):
        resp = ServerResponse()
        
        if 'reset' in req:
            self.orbital = orbitalcl.SimController_CL()
            self.clock = SimClock()
            resp.reset = True
        
        if 'load' in req:
            self.orbital.load_from_dict(req['load'])
            resp.load = True
         
        if 'device' in req:
            device = req['device']
            self.orbital.set_cl_device(device['platform_id'], device['device_id'])
            resp.device = DeviceResponse(**device)
        
        if req.get('init', False):
            if not self.orbital.is_initialized:
                self.orbital.init_sim(req["particles"])
                resp.init = True
                resp.config = ConfigResponse(
                    G = self.orbital.G,
                    N = self.orbital.N,
                    coef_of_restitution=self.orbital.coef_of_restitution,
                    dt_base=self.orbital.dt_base,
                    EPS_DIST=self.orbital.EPS_DIST,
                    EPS_TIME=self.orbital.EPS_TIME
                )
            resp.clock = self.clock
            resp.memory = self._get_shared_memory_state()
        
        if req.get('sync', False):
            self.orbital.sync()
            resp.sync = True
            resp.clock = self.clock
        
        if req.get('start', False):
            self.tick_ctl = TickController(self.orbital, self.clock)
            self.tick_ctl.start()
            self.clock.start()
            resp.start = True
        
        if req.get('stop', False):
            if hasattr(self, 'tick_ctl'):
                self.tick_ctl.stop()
            self.clock.stop()
            resp.stop = True
            resp.clock = self.clock
        
        if 'clock' in req:
            self.clock.update(req['clock'])
            resp.clock = self.clock
        
        if 'tick' in req:
            n_ticks = self.orbital.tick_dt(req['tick'])
            resp.tick = n_ticks > 0
        
        if self.orbital.is_initialized:
            resp.status = StatusResponse(
                tick_id = self.orbital.tick_id,
                accum = float(self.orbital.accum)
            )
        
        return resp

if __name__ == "__main__":
    server = OrbitalControlServer()
    server.serve()