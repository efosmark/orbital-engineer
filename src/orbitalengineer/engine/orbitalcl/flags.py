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
        if a.upper() == a:
            value = getattr(current_module, a)
            defs.append((a, value))
    return sorted(defs, key=lambda x:x[1])

def _cl_flag_defs():
    import importlib
    module = importlib.import_module(__name__)
    lines = []
    defs = _get_definitions(orbitalengineer.flags)
    
    max_len_def_name = max(len(x[0]) for x in defs)
    for a,value in defs:
        lines.append(f"#define {a:<{max_len_def_name}} ((uint) 0b{value:032b})")
    
    return lines

def generate_cl_flag_file():
    lines = _cl_flag_defs()
    return _flag_header_template.strip().format(values="\n".join(lines))

if __name__ == "__main__":
    print(generate_cl_flag_file())