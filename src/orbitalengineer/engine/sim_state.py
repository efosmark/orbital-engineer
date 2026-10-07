

class SimState:
    N:int = 0
    
    tick_id:int = 0
    step_id:int = 0
    
    @property
    def run_id(self) -> tuple[int,int]:
        return (self.tick_id, self.step_id)