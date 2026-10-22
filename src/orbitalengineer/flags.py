
REMOVED             = 1
FIXED_VELOCITY      = 1 << 1
FIXED_POSITION      = 1 << 2
FIXED_MASS          = 1 << 3
FIXED_RADIUS        = 1 << 4
REPEL_ON_OVERLAP    = 1 << 5
BOUNCE_AS_PRIMARY   = 1 << 6
BOUNCE_AS_SECONDARY = 1 << 7
MERGE_AS_PRIMARY    = 1 << 8
MERGE_AS_SECONDARY  = 1 << 9

MERGE = MERGE_AS_PRIMARY|MERGE_AS_SECONDARY

BOUNCE = BOUNCE_AS_PRIMARY|BOUNCE_AS_SECONDARY

PRIMARY_BODY = FIXED_POSITION|FIXED_VELOCITY|FIXED_RADIUS|MERGE_AS_PRIMARY


########################3


import orbitalengineer

_flag_header_template = """
#pragma once

// Hey, you!
// This file is auto-generated. Do not edit directly.

{values}
"""

def _get_definitions(current_module):    
    defs:list[tuple[str,int]] = []
    for a in dir(current_module):
        if a.startswith('_'):continue
        if a.upper() == a:
            value = getattr(current_module, a)
            defs.append((a, value))
    return sorted(defs, key=lambda x:x[1])

def _cl_flag_defs():
    defs = _get_definitions(orbitalengineer.flags)
    max_len_def_name = max(len(x[0]) for x in defs)
    lines = []
    for a,value in defs:
        lines.append(f"#define {a:<{max_len_def_name}} ((uint) 0b{value:032b})")
    return lines

def generate_cl_flag_file():
    lines = _cl_flag_defs()
    return _flag_header_template.strip().format(values="\n".join(lines))

if __name__ == "__main__":
    print(generate_cl_flag_file())