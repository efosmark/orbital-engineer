from dataclasses import dataclass
import time
from typing import Any
from orbitalengineer.ui.gtk4 import GObject

DISPLAY_DURATION = 15.0

@dataclass
class LedgerEntry:
    id:int
    i:int
    j:int
    action:int
    start:float|None = None
    tick_id:int = 0
    step_id:int = 0
    
    @classmethod
    def from_dtype(cls, ledger_dtype):
        return cls(
            id=ledger_dtype['id'],
            i=ledger_dtype['i'],
            j=ledger_dtype['j'],
            action=ledger_dtype['action'],
            tick_id=ledger_dtype['tick_id'],
            step_id=ledger_dtype['step_id'],
        )
        
    def __str__(self):
        return f'[EVENT  id={self.id}  (i,j)=({self.i}, {self.j})  action={self.action}  t=({self.tick_id}, {self.step_id})]'

class LedgerModel(GObject.GObject):
    props:Any
    
    entries = GObject.Property(type=object) # LedgerEntry[]
    last_ledger_id = GObject.Property(type=int)
    last_tick_checked = GObject.Property(type=int)
    
    def __init__(self):
        super().__init__()
        self.props.entries = []
        self.last_ledger_id = 0
        self.last_tick_checked = 0

    def add_entry(self, entry):
        le = LedgerEntry.from_dtype(entry)
        le.start = time.monotonic()
        self.entries.append(le)
        self.notify('entries')
    
    def prune(self):
        self.entries = [e for e in self.entries if e.start + DISPLAY_DURATION >= time.monotonic()]
        self.notify('entries')
    
    def get_entries(self):
        self.prune()
        return self.entries
