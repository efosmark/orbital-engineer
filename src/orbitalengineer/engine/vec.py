import math
from typing import Optional, Tuple


Vec2 = Tuple[float, float]

# ---------------------------
# basic 2D vector operations
# ---------------------------

def v_add(a: Vec2, b: Vec2) -> Vec2:
    """Elementwise a + b."""
    return (a[0] + b[0], a[1] + b[1])

def v_sub(a: Vec2, b: Vec2) -> Vec2:
    """Elementwise a - b (vector from b to a)."""
    return (a[0] - b[0], a[1] - b[1])

def v_dot(a: Vec2, b: Vec2) -> float:
    """Dot product a·b."""
    return a[0]*b[0] + a[1]*b[1]

def v_scale(a: Vec2, s: float) -> Vec2:
    """Scale vector a by scalar s."""
    return (a[0]*s, a[1]*s)

def v_norm(a: Vec2) -> float:
    """Euclidean norm |a|."""
    return math.hypot(a[0], a[1])

def v_unit(a: Vec2, eps: float = 1e-15) -> Optional[Vec2]:
    """
    Unit vector in the direction of a.
    Returns None if |a| is too small (to avoid divide-by-zero).
    """
    n = v_norm(a)
    if n < eps:
        return None
    return (a[0]/n, a[1]/n)