import math

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


def mag_format(x, sig=1):
    if abs(x) < 1e-6:
        return f"0"
    elif abs(x) <= 1:
        return f"{x:.2f}"
    elif abs(x) <= 1000:
        return f"{x:.0f}"
    
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    mant = round(x / (10 ** exp), sig)
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