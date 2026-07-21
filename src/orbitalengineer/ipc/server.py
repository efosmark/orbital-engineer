import socket

from orbitalengineer.engine import logger

from orbitalengineer.engine.clock import SimClock
from orbitalengineer.engine.config import SERVER_IPC_HOST, SERVER_IPC_PORT
from orbitalengineer.engine.orbitalcl import orbitalcl
from orbitalengineer.ipc import message, transport
from orbitalengineer.ipc.ticker import TickController

import pyopencl as cl
import numpy as np

class OrbitalControlServer:
    tick_ctl:TickController

    def __init__(self):
        self.orbital = orbitalcl.SimController_CL()
        self.clock = SimClock()

    def serve(self, host=SERVER_IPC_HOST, port=SERVER_IPC_PORT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(1)
            print(f"Orbital Server started. {host=} {port=}")
            conn, addr = s.accept()
            try:
                self._handle_client(conn, addr)
            except ConnectionResetError as e:
                print("Connection reset by peer.")
        self.orbital.disconnect()

    def initialize(self, platform_id:int, device_id:int, particles):
        if self.orbital.is_initialized:
            logger.warning("Already initialized. Re-initializing...")
            self.end()
        self.orbital.set_cl_device(platform_id, device_id)
        self.orbital.init_sim(particles)

    def start(self):
        self.tick_ctl = TickController(self.orbital, self.clock)
        self.tick_ctl.start()
        self.clock.start()

    def pause(self):
        if hasattr(self, 'tick_ctl'):
            self.tick_ctl.stop()
        self.clock.stop()
    
    def end(self):
        self.pause()
        self.orbital.disconnect()
        self.orbital.is_initialized = False

    def _get_shared_memory_info(self, field) -> message.SharedMemoryInfo:
        return message.SharedMemoryInfo(
            name=self.orbital.shm[field].name,
            dtype=str(getattr(self.orbital, field).dtype),
            size=self.orbital.shm[field].size,
            shape=getattr(self.orbital, field).shape
        )

    def _get_init_response(self):            
        return message.InitResponse(
            initialized=self.orbital.is_initialized,
            config=self._get_config_response(),
            memory=self._get_shared_memory_response()
        )

    def _get_config_response(self):
        return message.ConfigResponse(
            G = self.orbital.G,
            N = self.orbital.N,
            coef_of_restitution=self.orbital.coef_of_restitution,
            dt_base=self.orbital.dt_base,
            EPS_DIST=self.orbital.EPS_DIST,
            EPS_TIME=self.orbital.EPS_TIME
        )

    def _get_shared_memory_response(self):
        return message.SharedMemoryResponse(
            flags=self._get_shared_memory_info('flags'),
            velocity=self._get_shared_memory_info('velocity'),
            position=self._get_shared_memory_info('position'),
            mass=self._get_shared_memory_info('mass'),
            radius=self._get_shared_memory_info('radius'),
            force=self._get_shared_memory_info('force'),
        )

    def _get_status_response(self):
        return message.StatusResponse(
            initialized=self.orbital.is_initialized,
            tick_id=self.orbital.tick_id,
            accum=float(self.orbital.accum),
            clock=self.clock,
        )

    def _handle_client(self, conn, addr):
        print("Connection from", addr)
        try:
            while True:
                try:
                    _, message_type, payload = transport.recv_message(conn, message.MessageType)
                except ConnectionError as e:
                    logger.error("Connection error: %s", e)
                    break
                self._handle_request(conn, message.MessageType(message_type), payload)
        except KeyboardInterrupt:
            print("Shutting down server.")
        self.end()

    def _handle_request(self, conn:socket.socket, message_type:message.MessageType, payload):
        if message_type == message.MessageType.INIT_REQ:
            req = message.InitRequest.from_dict(payload)
            self.initialize(req.device_id, req.platform_id, req.particles)
            transport.send_message(conn, message.MessageType.INIT_RESP, self._get_init_response())
        
        elif message_type == message.MessageType.SYNC_REQ:
            self.orbital.sync()
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())

        elif message_type == message.MessageType.STATUS_REQ:
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
        
        elif message_type == message.MessageType.SHIFT_VECTOR_REQ:
            req = message.ShiftVectorsRequest(**payload)
            self.orbital.apply_vector_offset(req.vector_name, req.ids, req.op, req.offset)
            self.orbital.nudge()
            transport.send_message(conn, message.MessageType.SUCCESS)
        
        elif message_type == message.MessageType.CLOCK_UPDATE:
            req = message.ClockUpdateRequest.from_dict(payload)
            
            if req.running is not None:
                if req.running:
                    self.start()
                else:
                    self.pause()
            
            if req.speed is not None:
                self.clock.speed = req.speed
                
            transport.send_message(conn, message.MessageType.SUCCESS)
        
        elif message_type == message.MessageType.END_REQ:
            self.end()
            transport.send_message(conn, message.MessageType.SUCCESS)
        
        else:
            logger.error("Unrecognized message_type %s", message_type)

if __name__ == "__main__":
    server = OrbitalControlServer()
    server.serve()