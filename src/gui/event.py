from gui.model import engine, ledger
from gui.gtk4 import Gtk, Gio, GObject, GLib

from orbitalengineer import flags

class Event(GObject.GObject):

    tick = GObject.Signal(name='tick', arg_types=(int,),)
    bouncing = GObject.Signal(name='bouncing', arg_types=(object,),)
    merging = GObject.Signal(name='merging', arg_types=(object,),)
    
    def __init__(self, engine_model:engine.EngineModel, ledger_model:ledger.LedgerModel):
        self.engine_model = engine_model
        self.ledger_model = ledger_model
        self.engine_model.connect('notify::tick-id', self._search_for_ledger_entries)
        self.engine_model.connect('notify::tick-id', self._emit_tick_changed)

    def _emit_tick_changed(self, m, param):
        self.emit("tick", self.engine_model.tick_id)
    
    def _search_for_ledger_entries(self, _model, param):
        if self.engine_model.ledger is None or self.ledger_model.last_tick_checked == self.engine_model.tick_id:
            return

        ledger_size = self.engine_model.config.LEDGER_SIZE
        prev_ledger_id = self.ledger_model.last_ledger_id
        
        for i in range(1, ledger_size + 1):
            idx = (prev_ledger_id + i) % ledger_size
            le = self.engine_model.ledger[idx]
            if le['id'] > self.ledger_model.last_ledger_id:
                entry = self.ledger_model.add_entry(le)
                self.ledger_model.last_ledger_id = le['id']
                self.emit('new-entry', entry)

            elif le['id'] <= prev_ledger_id:
                # Rolled over to older values
                break
        
        self.ledger_model.last_tick_checked = self.engine_model.tick_id


    def _on_ledger_entry_added(self, m, le):
        if le.action&flags.BOUNCE:
            self.emit('bouncing', le)
        elif (le.action&flags.MERGE_AS_PRIMARY) or (le.action&flags.MERGE_AS_SECONDARY):
            self.emit('merging', le)
