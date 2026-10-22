import numpy as np

import orbitalengineer
from orbitalengineer.engine import config

COMPLEX = np.complex64
REAL = np.float32

if config.USE_DOUBLE_PRECISION:
    COMPLEX = np.complex128
    REAL = np.float64


_dtype_header_template = """
#pragma once

// Hey, you!
// This file is auto-generated. Do not edit directly.

{values}

#define COMPLEX_WG_REDUCE_ADD(T, var) ((T)(work_group_reduce_add(var.x), work_group_reduce_add(var.y)))

"""

_NP_TO_CL_DTYPE = {
    np.complex64: 'float2',
    np.complex128: 'double2',
    np.float32: 'float',
    np.float64: 'double'
}

def _get_definitions(current_module):    
    defs:list[tuple[str,int]] = []
    for a in dir(current_module):
        if a.startswith('_'): continue
        if a.upper() == a:
            value = getattr(current_module, a)
            defs.append((a, value))
    return defs

def _cl_dtype_defs():
    defs = _get_definitions(orbitalengineer.dtypes)
    
    max_len_def_name = max(len(x[0]) for x in defs)
    lines = []
    for a,value in defs:
        if value not in _NP_TO_CL_DTYPE: raise Exception("what", value)
        lines.append(f"#define {a:<{max_len_def_name}} {_NP_TO_CL_DTYPE[value]}")
    return lines

def generate_cl_dtype_file():
    lines = _cl_dtype_defs()
    return _dtype_header_template.strip().format(values="\n".join(lines))

if __name__ == "__main__":
    print(generate_cl_dtype_file())