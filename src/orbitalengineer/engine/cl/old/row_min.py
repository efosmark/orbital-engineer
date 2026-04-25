# import pyopencl as cl
# import numpy as np

# from orbitalengineer.engine.cl.kernelizer import KernelBuilder


# kernel_src = """
# __kernel void global_row_reduce(
#              const float  min_val,
#              const float  max_val,
#     __global const float* partials,
#     __global float *restrict result
# ) {
#     uint N  = get_global_size(0);
#     uint Lx = get_local_size(1);

#     uint row = get_global_id(0);
#     uint g1 = get_global_id(1);

#     // Work-item grid stride
#     float value = INFINITY;
#     uint base = row * N;
    
#     for (uint idx = base + g1; idx < base + N; idx += Lx) {
#         if (partials[idx] > min_val)
#             value = fmin(value, partials[idx]);
#     }
    
#     // Sub-group reduction
#     float sg_min = sub_group_reduce_min(value);
#     if (get_sub_group_local_id() != 0) {
#         sg_min = INFINITY;
#     }

#     // Work-group reduction
#     float total_min = work_group_reduce_min(sg_min);

#     // Work-group leader writes result
#     if (get_local_id(0) == 0 && get_local_id(1) == 0) {
#         result[row] = total_min;
#     }
# }
# """


# class RowReducerMin(KernelBuilder):
#     kernel_name = "global_row_reduce"
#     kernel_src = kernel_src
    
#     def __call__(self, N:int, min_val, max_val, input_table:cl.Buffer, output_buffer:cl.Buffer, *, local_size:int=256):
#         Lx = int(local_size) 
#         return self.kernel(
#             self.queue,
#             (N, Lx),     # global work size
#             (1, Lx),     # local work size
            
#             # Args
#             np.float32(min_val),
#             np.float32(max_val),
#             input_table,
#             output_buffer,
#         )