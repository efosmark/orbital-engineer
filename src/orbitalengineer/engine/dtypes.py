import numpy as np

import orbitalengineer

_dtype_header_template = """
#pragma once

// Hey, you!
// This file is auto-generated. Do not edit directly.

{values}


#define COMPLEX_WG_REDUCE_ADD(T, var) ((T)(work_group_reduce_add(var.x), work_group_reduce_add(var.y)))

"""

NP_TO_CL_DTYPE = {
    np.complex64: 'float2',
    np.complex128: 'double2',
    np.float32: 'float',
    np.float64: 'double'
}

def _get_definitions(current_module):    
    defs:list[tuple[str,int]] = []
    for a in dir(current_module):
        if a.upper() == a:
            value = getattr(current_module, a)
            defs.append((a, value))
    return defs

def _cl_dtype_defs():
    defs = _get_definitions(orbitalengineer.dtypes)
    
    max_len_def_name = max(len(x[0]) for x in defs)
    lines = []
    for a,value in defs:
        
        if value not in NP_TO_CL_DTYPE:
            raise Exception("what", value)
        
        lines.append(f"#define {a:<{max_len_def_name}} {NP_TO_CL_DTYPE[value]}")
    
    return lines

def generate_cl_dtype_file():
    lines = _cl_dtype_defs()
    return _dtype_header_template.strip().format(values="\n".join(lines))

if __name__ == "__main__":
    print(generate_cl_dtype_file())