import socket

from orbitalengineer.engine import logger
from orbitalengineer.engine.orbitalcl import orbitalcl
from orbitalengineer.ipc import message, transport
from orbitalengineer.ipc.ticker import TickController
from orbitalengineer.ipc.clock import SimClock
from orbitalengineer.ipc.config import SERVER_IPC_HOST, SERVER_IPC_PORT

class OrbitalControlServer:
    tick_ctl:TickController

    def __init__(self):
        self.orbital = orbitalcl.SimController_CL()
        self.clock = SimClock()
        self.tick_ctl = TickController(self.orbital, self.clock)
        self.enabled = True

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

    def initialize(self, particles):
        if self.orbital.is_initialized:
            logger.warning("Already initialized. Re-initializing...")
            self.end()
            self.orbital.reset()
        self.enabled = True
        self.orbital.set_cl_device(self.device.platform_id, self.device.device_id)
        if self.orbital.init_sim(particles):
            self.clock.reset()
            self.tick_ctl.reset()
            logger.info("Initialized.")
            return True
        else:
            logger.error("Failed to initialize.")
            return False

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
            dtype=str(vec.dtype),
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
            flags=self._get_shared_memory_info('flags'),
            velocity=self._get_shared_memory_info('velocity'),
            position=self._get_shared_memory_info('position'),
            mass=self._get_shared_memory_info('mass'),
            radius=self._get_shared_memory_info('radius'),
            force=self._get_shared_memory_info('force'),
            #cgroup=self._get_shared_memory_info('cgroup'),
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
            gpu_status=gpu_status
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
            result = self.initialize(req.particles)
            if not result:
                transport.send_message(conn, message.MessageType.ERROR, message.ErrorResponse(False, "Unable to initialize."))
                return
            transport.send_message(conn, message.MessageType.INIT_RESP, self._get_init_response())
        
        elif message_type == message.MessageType.SYNC_REQ:
            self.orbital.state.sync()
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())

        elif message_type == message.MessageType.STATUS_REQ:
            transport.send_message(conn, message.MessageType.STATUS_RESP, self._get_status_response())
        
        elif message_type == message.MessageType.SHIFT_VECTOR_REQ:
            req = message.ShiftVectorsRequest(**payload)
            self.orbital.state.apply_vector_offset(req.vector_name, req.ids, req.op, req.offset)
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