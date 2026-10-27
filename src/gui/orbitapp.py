from gui import ui_config
from gui.controller.client_sync import ClientSyncController
from gui.controller.ledger_ctl import LedgerMonitor
from gui.model import engine, ledger
from gui.gtk4 import Gtk, Gio, GObject, GLib

from orbitalengineer import flags, body
from orbitalengineer.engine import logger
from orbitalengineer.ipc.client import ClientSocketConnection


class OrbitApp(Gtk.Application):
    platform_id = GObject.Property(type=int, default=-1)
    device_id = GObject.Property(type=int, default=-1)
    
    _initialize = GObject.Signal(name='initialize')
    _sync = GObject.Signal(name='sync')
    _tick = GObject.Signal(name='tick', arg_types=(int,),)
    _bouncing = GObject.Signal(name='bouncing', arg_types=(object,),)
    _merging = GObject.Signal(name='merging', arg_types=(object,),)
    
    def __init__(self, platform_id=-1, device_id=-1, reset:bool=False):
        super().__init__(application_id=ui_config.APP_ID, flags=Gio.ApplicationFlags.NON_UNIQUE)        
        self._should_reset_on_connect = reset        
        self.platform_id = platform_id
        self.device_id = device_id
        
        self.client = ClientSocketConnection()
        self.client.set_device(platform_id, device_id)

        self.ledger_model = ledger.LedgerModel()
        self.engine_model = engine.EngineModel()
        self.sync_ctl = ClientSyncController(self.client, self.engine_model)
        
        self.ledger_ctl = LedgerMonitor(self.engine_model, self.ledger_model)
        self.ledger_ctl.connect('new-entry', self.on_ledger_entry_added)

        self.engine_model.connect("notify::tick-id", self.__on_sim_tick_changed)
        self.engine_model.connect("notify::clock-running", self.__on_clock_running_changed)

        self.connect('activate', self.__on_activate)

    def __on_sim_tick_changed(self, m, tick_id):
       self.emit("tick", self.engine_model.tick_id)

    def __on_clock_running_changed(self, m, prop):
        if self.engine_model.clock_running:
            self.client.start()
        else:
            self.client.stop()

    def on_ledger_entry_added(self, m, le):
        if le.action&flags.BOUNCE:
            self.emit('bouncing', le)
        elif (le.action&flags.MERGE_AS_PRIMARY) or (le.action&flags.MERGE_AS_SECONDARY):
            self.emit('merging', le)

    def _bootstrap(self):
        self.client.connect(reset=self._should_reset_on_connect)
        self.client.sync_full_state()
        self.sync_ctl.sync()
        
    def __on_activate(self, app):
        self._bootstrap()
        if not self.engine_model.is_initialized or self._should_reset_on_connect:
            self.emit('initialize')

    def insert_particle(self, particle:body.Body) -> int:
        return self.client.add_particle(
            position=particle.position,
            velocity=particle.velocity,
            mass=particle.mass,
            radius=particle.radius,
            flags=particle.flags
        )

    def tick_once(self):
        if not hasattr(self, '_last_tick'):
            self._last_tick = None
        if self._last_tick is None or self.client.tick_id > self._last_tick:
            self._last_tick = self.client.tick_id
            GLib.idle_add(self.client.tick_once)

    def substep_once(self):
        self.client.substep_once()
