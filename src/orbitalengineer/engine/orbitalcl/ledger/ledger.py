import numpy as np
import pyopencl as cl

from orbitalengineer.engine.config import LEDGER_SIZE
from orbitalengineer.engine.orbitalcl.dimension import PipelineComponent

ledger_dtype = np.dtype([
    ("id",     np.uint32), 
    ("i",      np.uint32),
    ("j",      np.uint32), 
    ("action", np.uint32), 
    ("tick_id", np.uint32), 
    ("step_id", np.uint32),
], align=True)

KERNEL_FILE_LOCATION = "ledger/ledger.cl"

class LedgerController(PipelineComponent):
    debug_flag = "ledger"
    
    ledger:cl.Buffer
    last_ledger_entry_committed:int = 0

    def initialize(self):
        self._commit_ledger = self._load_kernel("commit_ledger", KERNEL_FILE_LOCATION)
        self.ledger_entry_count = self.alloc(1, dtype=np.uint32)
        self.ledger = self.alloc(LEDGER_SIZE, dtype=ledger_dtype, shared_name='ledger')
    
    def commit(self, tick_id:int, step_id:int):
        ledger_entry_count = self.get_host_vector(self.ledger_entry_count, sync=True)[0]
        
        num_entries = ledger_entry_count - self.last_ledger_entry_committed
        if num_entries == 0: return
        
        evt = self._commit_ledger(
                self.queue,
                (num_entries + 1, ),
                None,
                
                # Args
                np.uint32(self.last_ledger_entry_committed),
                np.uint32(tick_id),
                np.uint32(step_id),
                self.ledger,
            )
        
        self.last_ledger_entry_committed += num_entries - 1
        return evt
    
    def sync(self):
        self.sync_to_host(self.ledger).wait()
