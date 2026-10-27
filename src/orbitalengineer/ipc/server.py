import os
CPU_NUM = 15 # TODO: instead of manually pinning, identify the most appropriate core to pin to.
os.sched_setaffinity(0, {CPU_NUM})

import socket
import select
import time

import psutil

from orbitalengineer.engine import logger
from orbitalengineer.engine import controller
from orbitalengineer.engine.sim_config import SimConfig
from orbitalengineer.ipc import message, transport
from orbitalengineer.ipc.ticker import TickController
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.config import SERVER_IPC_HOST, SERVER_IPC_PORT, SERVER_BACKLOG_SIZE
from orbitalengineer.ipc.message import MessageType

class OrbitalControlServer:
    host_status:message.HostStatus
    host_status_last_time:float = 0

    def __init__(self, host:str, port:int):
        self.host = host
        self.port = port
        self.enabled = True        
        self.orbital = controller.SimController()
        self.clock = SimClock()
        self.tick_ctl = TickController(self.orbital, self.clock)
        self.cpu_num = psutil.Process().cpu_num()
        if self.cpu_num != CPU_NUM:
            logger.warning("ERROR: set_affinity failed.   cpu_num=%s   CPU_NUM=%s", self.cpu_num, CPU_NUM)

        self._dispatch_map = {
            MessageType.INIT_REQ: self._handle_init_req,
            MessageType.RESET: self._handle_reset_req,
            MessageType.SYNC_REQ: self._handle_sync,
            MessageType.STATUS_REQ: self._handle_status,
            MessageType.CLOCK_UPDATE: self._handle_clock_update,
            MessageType.STATE_REQ: self._handle_state_req,
            MessageType.SHIFT_VECTOR_REQ: self._handle_shift_vector_req,
            MessageType.TICK_ONCE: self._handle_tick_once,
            MessageType.SUBSTEP_ONCE: self._handle_substep_once,
        }

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
        self.host_status = message.HostStatus(self.orbital.cpu_num, self.orbital.get_cpu_utilization())

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
            config=self.orbital.cfg,
            memory=self._get_shared_memory_response()
        )

    def _get_shared_memory_response(self):
        return message.SharedMemoryResponse(
            id_to_index=self._get_shared_memory_info('id_to_index'),
            body_id=self._get_shared_memory_info('body_id'),
            flags=self._get_shared_memory_info('flags'),
            velocity=self._get_shared_memory_info('velocity'),
            position_A=self._get_shared_memory_info('position_A'),
            position_B=self._get_shared_memory_info('position_B'),
            mass=self._get_shared_memory_info('mass'),
            radius=self._get_shared_memory_info('radius'),
            force=self._get_shared_memory_info('force'),
            ledger=self._get_shared_memory_info('ledger'),
            cgroup=self._get_shared_memory_info('cgroup'),
            n_direct_contacts=self._get_shared_memory_info('n_direct_contacts'),
            direct_contacts=self._get_shared_memory_info('direct_contacts'),
            kinetic_energy=self._get_shared_memory_info('kinetic_energy'),
            grav_potential_energy=self._get_shared_memory_info('grav_potential_energy'),
        )

    def _get_status_response(self):
        if hasattr(self.orbital, 'device'):
            gpu_status = self.orbital.device.gpu_status()
        else:
            gpu_status = message.DeviceStatus((-1, -1), 'n/a', 'n/a', 0, 0, 0)
        
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
            host_status=self.get_cpu_status(),
            buffer=self.orbital.vec.buffer_id if self.orbital.is_initialized else 'A'
        )
    
    def _handle_state_req(self, conn, payload):
        state_resp = message.StateResponse(
            status=self._get_status_response(),
            config=self.orbital.cfg if self.orbital.is_initialized else None,
            memory=self._get_shared_memory_response() if self.orbital.is_initialized else None
        )
        transport.send_message(conn, MessageType.STATE_RESP, state_resp)
    
    def _handle_clock_update(self, conn, payload):
        req = message.ClockUpdateRequest.from_dict(payload)
        if req.running is not None:
            if req.running: self.start()
            else:           self.pause()
        if req.speed is not None:
            self.clock.speed = req.speed
        transport.send_message(conn, MessageType.SUCCESS)

    def _handle_init_req(self, conn, payload):
        req = message.InitRequest.from_dict(payload)
        self.device = req.device
        result = self.initialize(req.particles, req.config)
        if not result:
            transport.send_message(conn, MessageType.ERROR, message.ErrorResponse(False, "Unable to initialize."))
        transport.send_message(conn, MessageType.INIT_RESP, self._get_init_response())

    def _handle_reset_req(self, conn, payload):
        self.orbital.reset()
        self._handle_status(conn)

    def _handle_sync(self, conn, payload):
        req = message.SyncRequest(**payload)
        self.orbital.sync(req.current_buffer)
        self._handle_status(conn)

    def _handle_status(self, conn, payload=None):
        transport.send_message(conn, MessageType.STATUS_RESP, self._get_status_response())

    def _handle_shift_vector_req(self, conn, payload):
        req = message.ShiftVectorsRequest(**payload)
        self.orbital.vec.apply_vector_offset(req.vector_name, req.ids, req.op, req.offset)
        self.orbital.nudge()
        transport.send_message(conn, MessageType.SUCCESS, None)

    def _handle_tick_once(self, conn, payload):
        self.pause()
        self.clock.increment_by(self.orbital.cfg.DEFAULT_DT_BASE)
        self.orbital.tick(self.orbital.cfg.DEFAULT_DT_BASE)
        transport.send_message(conn, MessageType.STATUS_RESP, self._get_status_response())
    
    def _handle_substep_once(self, conn, payload):
        self.pause()
        dt = float(self.orbital.substep(self.orbital.cfg.DEFAULT_DT_BASE))
        self.clock.increment_by(dt)
        transport.send_message(conn, MessageType.STATUS_RESP, self._get_status_response())

    def _setup_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setblocking(False)
        self.server.bind((self.host, self.port))
        self.server.listen(SERVER_BACKLOG_SIZE)

    def _accept_new_connection(self, s:socket.socket):
        connection, client_address = s.accept()
        connection.setblocking(False)
        self.inputs.append(connection)
        print('New connection from', client_address)

    def _dispatch_readable(self, s:socket.socket):
        try:
            _, message_type, payload = transport.recv_message(s, MessageType)
            mtype = MessageType(message_type)
            
            if mtype != MessageType.SYNC_REQ:
                logger.debug(f"<== [{mtype.name}] {str(payload)[:80]}")
            
            if mtype in self._dispatch_map:
                self._dispatch_map[mtype](s, payload)
            else:
                logger.error("Unrecognized message_type %s", mtype)
        except ConnectionError as e:
            logger.error("Connection error: %s", e)
            self.inputs.remove(s)
            s.close()
            print("Connection closed.")

    def serve_forever(self):
        self._setup_server()
        self.inputs = [self.server]
        while True:
            readable, _wr, _ex = select.select(self.inputs, [], self.inputs)
            for s in readable:
                if s is self.server:
                    self._accept_new_connection(s)
                else:
                    self._dispatch_readable(s)

if __name__ == "__main__":
    server = OrbitalControlServer(SERVER_IPC_HOST, SERVER_IPC_PORT)
    
    print(f"Orbital Server started. {SERVER_IPC_HOST=} {SERVER_IPC_PORT=}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.orbital.shm.disconnect()
        print("Server shutdown.")        