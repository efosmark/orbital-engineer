from dataclasses import dataclass

from gui.gtk4 import GObject
from gui.model.ledger import LedgerModel
from gui.model.engine import EngineModel

@dataclass
class LedgerEntry:
    ledger_id: int
    i: int
    j: int
    action: int
    tick_id: int
    step_id: int

class LedgerMonitor(GObject.GObject):

    new_entry = GObject.Signal(name='new-entry', arg_types=(object,),)

    def __init__(self, engine_model:EngineModel, ledger_model:LedgerModel):
        GObject.GObject.__init__(self)
        self.engine_model = engine_model
        self.ledger_model = ledger_model
        self.engine_model.connect('notify::tick-id', self.on_tick_changed)
    
    def on_tick_changed(self, _model, param):
        if self.engine_model.ledger is None or self.ledger_model.last_tick_checked == self.engine_model.tick_id:
            return

        ledger_size = self.engine_model.config.LEDGER_SIZE
        prev_ledger_id = self.ledger_model.last_ledger_id
        
        for i in range(1, ledger_size + 1):
            idx = (prev_ledger_id + i) % ledger_size
            le = self.engine_model.ledger[idx]
            if le['id'] > self.ledger_model.last_ledger_id:
                self.ledger_model.add_entry(le)
                self.ledger_model.last_ledger_id = le['id']
                self.emit('new-entry', LedgerEntry(*le))

            elif le['id'] <= prev_ledger_id:
                # Rolled over to older values
                break
        
        self.ledger_model.last_tick_checked = self.engine_model.tick_id