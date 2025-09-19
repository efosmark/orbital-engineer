import numpy as np
import cairo

from orbitalengineer.engine.history import UNFOCUSED_SAMPLE_RATE
from orbitalengineer.engine.memory import BodyProxy, OrbitalMemory


def draw_body_history(cr:cairo.Context, shm:OrbitalMemory, b:BodyProxy, step_id, downsample=False, color=(0.3,0.3,0.3)):     # type: ignore
    history_idx = shm.history_index[b.idx]
    
    history:list = np.roll(shm.history[b.idx], -history_idx-1).tolist()
    history = [h for h in history if h != 0]
    
    if downsample:
        history = history[::UNFOCUSED_SAMPLE_RATE]
    
    cr.set_source_rgb(*color)

    last_position = None
    for position in history:
        
        if last_position is None:
            last_position = position
            cr.move_to(last_position.real, last_position.imag)
            continue
    
        cr.move_to(last_position.real, last_position.imag)
        cr.line_to(position.real, position.imag)
        cr.stroke()
        
        last_position = position
    
    cr.stroke()
    if last_position is not None:
        cr.move_to(last_position.real, last_position.imag)
        cr.line_to(*b.get_xy(step_id))
        cr.stroke()