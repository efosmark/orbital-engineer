from dataclasses import dataclass

from orbitalengineer.ui.gtk4 import GObject
from orbitalengineer.ui.model.ledger import LedgerModel
from orbitalengineer.ui.model.main import AppModel

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

    def __init__(self, app:AppModel, model:LedgerModel):
        GObject.GObject.__init__(self)
        self.app = app
        self.model = model
        self.app.engine.connect('notify::tick-id', self.on_tick_changed)
    
    def on_tick_changed(self, _model, param):
        if self.app.engine.ledger is None or self.model.last_tick_checked == self.app.engine.tick_id:
            return

        ledger_size = self.app.engine.config.LEDGER_SIZE
        prev_ledger_id = self.model.last_ledger_id
        
        for i in range(1, ledger_size + 1):
            idx = (prev_ledger_id + i) % ledger_size
            le = self.app.engine.ledger[idx]
            if le['id'] > self.model.last_ledger_id:
                self.model.add_entry(le)
                self.model.last_ledger_id = le['id']
                self.emit('new-entry', LedgerEntry(*le))

            elif le['id'] <= prev_ledger_id:
                # Rolled over to older values
                break
        
        self.model.last_tick_checked = self.app.engine.tick_id