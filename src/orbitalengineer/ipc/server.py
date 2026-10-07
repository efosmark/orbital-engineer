import os
CPU_NUM = 15 # TODO: instead of manually pinning, identify the most appropriate core to pin to.
os.sched_setaffinity(0, {CPU_NUM})

import socket
import time

import psutil

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalcl import orbitalcl
from orbitalengineer.engine.orbitalcl.sim_config import SimConfig
from orbitalengineer.ipc import message, transport
from orbitalengineer.ipc.ticker import TickController
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.config import SERVER_IPC_HOST, SERVER_IPC_PORT

class OrbitalControlServer:
    tick_ctl:TickController
    
    host_status:message.HostStatus
    host_status_last_time:float = 0

    def __init__(self):
        self.orbital = orbitalcl.SimController_CL()
        self.clock = SimClock()
        self.tick_ctl = TickController(self.orbital, self.clock)
        self.enabled = True
        
        self.cpu_num = psutil.Process().cpu_num()
        if self.cpu_num != CPU_NUM:
            logger.warning("ERROR: set_affinity failed.   cpu_num=%s   CPU_NUM=%s", self.cpu_num, CPU_NUM)

    def serve(self, host=SERVER_IPC_HOST, port=SERVER_IPC_PORT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"Orbital Server started. {host=} {port=}")
            while self.enabled:
                conn, addr = s.accept()
                try:
                    self._handle_client(conn, addr)
                except ConnectionResetError as e:
                    print("Connection reset by peer:", addr)
            self.end()

    def initialize(self, particles, config:SimConfig|None):        
        if self.orbital.is_initialized:
            logger.warning("Already initialized. Re-initializing with %s bodies...", len(particles))
            self.end()
            self.orbital.reset()
        self.enabled = True
        self.orbital.set_cl_device(self.device.platform_id, self.device.device_id)
        if config is None:
            config = SimConfig()
        self.orbital.cfg = config
        if self.orbital.init_sim(particles):
            self.clock.reset()
            self.tick_ctl.reset()
            logger.info("Initialized.")
            return True
        else:
            logger.error("Failed to initialize.")
            return False

    def _update_host_status(self):
        utilization = psutil.cpu_percent(interval=None, percpu=True)[self.cpu_num]
        self.host_status = message.HostStatus(self.cpu_num, utilization)

    def get_cpu_status(self) -> message.HostStatus:
        now = time.monotonic()
        if now - self.host_status_last_time > 0.5:
            self._update_host_status()
            self.host_status_last_time = now        
        return self.host_status

    def start(self):
        self.tick_ctl.start()
        self.clock.start()
        logger.debug("Started. tick_id=%.0f  time=%.2f", self.orbital.pipeline_state.tick_id, self.clock.time())

    def pause(self):
        self.tick_ctl.stop()
        self.clock.stop()
        logger.debug("Paused.  tick_id=%.0f  time=%.2f", self.orbital.pipeline_state.tick_id, self.clock.time())
    
    def end(self):
        logger.debug("Ending simulation.")
        self.pause()
        self.orbital.shm.disconnect()
        self.orbital.is_initialized = False
        self.enabled = False

    def _get_shared_memory_info(self, field) -> message.SharedMemoryInfo:
        vec = self.orbital.shm.vec.get(field)
        if vec is None:
            raise Exception(f"Shared memory {field} vector does not exist.")
        
        return message.SharedMemoryInfo(
            name=self.orbital.shm[field].name,
            dtype=vec.dtype.name if vec.dtype.isbuiltin else vec.dtype.descr,
            size=vec.size,
            shape=vec.shape
        )

    def _get_init_response(self):            
        return message.InitResponse(
            initialized=self.orbital.is_initialized,
            config=self._get_config_response(),
            memory=self._get_shared_memory_response()
        )

    def _get_config_response(self):
        return self.orbital.cfg

    def _get_shared_memory_response(self):
        return message.SharedMemoryResponse(
            id_to_index=self._get_shared_memory_info('id_to_index'),
            body_id=self._get_shared_memory_info('body_id'),
            flags=self._get_shared_memory_info('flags'),
            velocity=self._get_shared_memory_info('velocity'),
            position=self._get_shared_memory_info('position'),
            mass=self._get_shared_memory_info('mass'),
            radius=self._get_shared_memory_info('radius'),
            force=self._get_shared_memory_info('force'),
            ledger=self._get_shared_memory_info('ledger'),
            cgroup=self._get_shared_memory_info('cgroup'),
            n_direct_contacts=self._get_shared_memory_info('n_direct_contacts'),
            direct_contacts=self._get_shared_memory_info('direct_contacts'),
        )

    def _get_status_response(self):
        gpu_status = None
        if hasattr(self.orbital, 'device'):
            gpu_status = self.orbital.device.gpu_status()
        
        return message.StatusResponse(
            initialized=self.orbital.is_initialized,
            tick_id=self.orbital.pipeline_state.tick_id,
            N=self.orbital.pipeline_state.N,
            accum=float(self.tick_ctl.total_dt_lag) if hasattr(self, 'tick_ctl') else 0,
            clock=self.clock,
            max_speed=None,
            curr_tick_at=self.tick_ctl.curr_tick_at,
            next_tick_at=self.tick_ctl.next_tick_at,
            dt_step=self.tick_ctl.dt_step,
            device_status=gpu_status,
            host_status=self.get_cpu_status()
        )
    
    def _get_state_response(self):
        return message.StateResponse(
            status=self._get_status_response(),
            config=self._get_config_response() if self.orbital.is_initialized else None,
            memory=self._get_shared_memory_response() if self.orbital.is_initialized else None
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
                
                #with self.orbital.tr(f"server.handle_request({message.MessageType(message_type).name})"):
                r = self._handle_request(conn, message.MessageType(message_type), payload)
                #if r == False:
                #    break
        except KeyboardInterrupt:
            print("Shutting down server.")
        #self.pause()

    def _handle_request(self, conn:socket.socket, message_type:message.MessageType, payload):
        if message_type == message.MessageType.INIT_REQ:
            req = message.InitRequest.from_dict(payload)
            self.device = req.device
            result = self.initialize(req.particles, req.config)
            if not result:
                transport.send_message(conn, message.MessageType.ERROR, message.ErrorResponse(False, "Unable to initialize."))
                return
            transport.send_message(conn, message.MessageType.INIT_RESP, self._get_init_response())

        elif message_type == message.MessageType.RESET:
            self.orbital.reset()
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
        
        elif message_type == message.MessageType.SYNC_REQ:
            self.orbital.sync()
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())

        elif message_type == message.MessageType.STATUS_REQ:
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
        
        elif message_type == message.MessageType.SHIFT_VECTOR_REQ:
            req = message.ShiftVectorsRequest(**payload)
            self.orbital.vec.apply_vector_offset(req.vector_name, req.ids, req.op, req.offset)
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
        
        elif message_type == message.MessageType.STATE_REQ:
            transport.send_message(conn, message.MessageType.STATE_RESP, self._get_state_response())
                
        elif message_type == message.MessageType.END_REQ:
            self.end()
            transport.send_message(conn, message.MessageType.SUCCESS)
        
        elif message_type == message.MessageType.DISCONNECT:
            conn.close()
            logger.info("Client disconnected.")
            return False
        
        elif message_type == message.MessageType.TICK_ONCE:
            self.pause()
            self.clock.increment_by(self.orbital.cfg.DEFAULT_DT_BASE)
            self.orbital.tick(self.clock.time())
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
        
        elif message_type == message.MessageType.SUBSTEP_ONCE:
            self.pause()
            dt = float(self.orbital.substep(self.orbital.cfg.DEFAULT_DT_BASE))
            self.clock.increment_by(dt)
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
            
        else:
            logger.error("Unrecognized message_type %s", message_type)

if __name__ == "__main__":
    server = OrbitalControlServer()
    server.serve()