import numpy as np

def create_pair_partition_mask(num_pairs, partition_id, num_partitions):
    indices = np.arange(num_pairs)
    split_indices = np.array_split(indices, num_partitions)
    
    mask = np.zeros(num_pairs, dtype=bool)
    mask[split_indices[partition_id]] = True
    return mask
