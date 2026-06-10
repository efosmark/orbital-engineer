import math
import numpy as np

magnitudes = {
    -6: "u",
    -3: "m",
    1: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T"
}

def eng_format(x, sig=3):
    if x == 0:
        return f"{0:.{sig}f}"
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    mant = x / (10 ** exp)
    return f"{mant:.{sig}f}e{exp:+}"

def get_scale_value(z):
    if z < 1:
        z = -1/z
    return z

def get_exp(x, sig, places=3):
    exp = int(math.floor(math.log10(abs(x)) / places) * places)
    mant = round(x / (10 ** exp), sig)
    return exp, mant

def mag_format(x, sig=1, eps=1e-12):
    if abs(x) < eps:
        return f"0"
    elif abs(x) <= 1:
        return f"{x:.2f}"
    elif abs(x) <= 1000:
        return f"{x:.0f}"
    elif math.isnan(x):
        return 'NaN'
    
    exp, mant = get_exp(x, sig)
    if exp in magnitudes:
        return f"{mant:.{sig}f}{magnitudes[exp]}"
    return f"{mant:.{sig}f}e{exp:+}"

def format_size(size_bytes):
    unit = 1000
    suffixes = [' ', 'k', 'M', 'G']
    for suffix in suffixes:
        if size_bytes < unit:
            return f"{size_bytes:.1f} {suffix}"
        size_bytes /= unit
    return f"{size_bytes:.2f} {suffixes[-1]}"

def positive_angle(angle_rad: float) -> float:
    tau = 2 * np.pi
    angle_rad %= tau
    if angle_rad < 0:
        angle_rad += tau
    return angle_rad % tau