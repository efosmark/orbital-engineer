import numpy as np
from numpy.typing import NDArray
from numba import njit, void, complex128, int64

## The minimum number of degrees betweeen each history point.
## Subsequent points below this threshold will be omitted.
## Setting this too high will cause history to be too short
MIN_HISTORY_DEG_CHANGE = 2.25


## The maximum number of points of history to track.
## High values can reduce performance.
HISTORY_DEPTH = int(360 // MIN_HISTORY_DEG_CHANGE)


# When we want to show history, but we do not need precision,
# we will skip intermediate values. 
UNFOCUSED_SAMPLE_RATE = 3


@njit(void(int64[:], int64, complex128[:,:], complex128[:,:], int64[:]), cache=True, fastmath=True)
def compute_history(
    body_ids: NDArray[np.int64], 
    step_id: np.int64,
    position: NDArray[np.complex128],
    history: NDArray[np.complex128],
    history_index: NDArray[np.int64]
):
    step_offset = step_id % 2
    for b_id in body_ids:
        current_position = position[step_offset, b_id]
        
        hist_idx = history_index[b_id]
        next_hist_idx = (hist_idx + 1) % HISTORY_DEPTH
        
        prev2 = history[b_id, (hist_idx-1) % HISTORY_DEPTH]
        prev1 = history[b_id, hist_idx]
        
        prev_rel = prev2 - prev1
        curr_rel = prev2 - current_position
        
        prev_angle = np.atan2(prev_rel.imag, prev_rel.real)
        curr_angle = np.atan2(curr_rel.imag, curr_rel.real)
        
        if np.degrees(np.abs(curr_angle - prev_angle)) > MIN_HISTORY_DEG_CHANGE:
            history[b_id, next_hist_idx] = current_position
            history_index[b_id] = next_hist_idx
        