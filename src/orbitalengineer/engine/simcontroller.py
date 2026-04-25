from typing import Any, Callable, Generator, Protocol
from orbitalengineer.engine.particle import Particle

StatusHandler_T = Callable[[int,int,int], bool|None]
MergeHandler_T = Callable[[int,int], bool|None]
CollisionHandler_T = Callable[[int,int], bool|None]

class OrbitalSimController(Protocol):
    N:int
    speed:float
    dt_base:float
    accum:float
    
    tick_id:int = 0
    step_count:int = 0
    
    is_initialized:bool = False
 
    def add_particle(self, particle:Particle) -> int:...
    
    def find_bodies_at(self, x:float, y:float, margin:float=10) -> Any:...

    def __iter__(self) -> Generator[Particle, Any, None]:...

    def get_valid_indices(self) -> Any:...

    def init_sim(self):...

    def get_particle(self, particle_id) -> Particle: ...
    
    def handle_interaction(self, body_i: int, body_j: int, prev: int, next: int):...

    # def add_merge_changed_handler(self, handler:MergeHandler_T):...

    # def add_collision_changed_handler(self, handler:CollisionHandler_T):...

    # def add_status_changed_handler(self, handler:StatusHandler_T):...

    def tick(self, now) -> Any: ...
    
    def sync(self): ...