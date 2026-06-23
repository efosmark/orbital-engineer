
_inc = 0
def auto():
    global _inc
    _inc += 1
    return 1 << _inc


REMOVED = auto()

FIXED_VELOCITY = auto()
FIXED_POSITION = auto()
FIXED_MASS     = auto()
FIXED_RADIUS   = auto()

REPEL_ON_OVERLAP = auto()

BOUNCE_AS_PRIMARY   = auto()
BOUNCE_AS_SECONDARY = auto()

MERGE_AS_PRIMARY = auto()
MERGE_AS_SECONDARY = auto()

MERGE = MERGE_AS_PRIMARY|MERGE_AS_SECONDARY
BOUNCE = BOUNCE_AS_PRIMARY|BOUNCE_AS_SECONDARY

PRIMARY_BODY = FIXED_POSITION|FIXED_VELOCITY|FIXED_RADIUS|MERGE_AS_PRIMARY

_flag_header_template = """
#pragma once

// Hey, you!
// This file is auto-generated. Do not edit directly.

{values}
"""

def _get_definitions(current_module):    
    defs:list[tuple[str,int]] = []
    for a in dir(current_module):
        if a.upper() == a:
            value = getattr(current_module, a)
            defs.append((a, value))
    return sorted(defs, key=lambda x:x[1])

def _cl_flag_defs():
    import importlib
    module = importlib.import_module(__name__)
    lines = []
    defs = _get_definitions(module)
    
    max_len_def_name = max(len(x[0]) for x in defs)
    for a,value in defs:
        lines.append(f"#define {a:<{max_len_def_name}} ((uint) 0b{value:032b})")
    
    return lines

def generate_cl_flag_file():
    lines = _cl_flag_defs()
    return _flag_header_template.strip().format(values="\n".join(lines))

if __name__ == "__main__":
    print(generate_cl_flag_file())