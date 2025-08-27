"""
Example: Custom Communication Primitives using CuTe DSL

This example shows how to implement custom communication operations
using the CuTe DSL and NVVM intrinsics.
"""

import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op, Boolean, Int32, if_generate
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector

from cutlass.cute.arch.nvvm_wrappers import (
    warp_sync_op,
    shuffle_sync_op,
    fence_acq_rel_cta,
    fence_acq_rel_cluster,
    thread_idx,
    warp_id,
    lane_id,
)

# Communication patterns
from enum import Enum

class CommPattern(Enum):
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    BROADCAST = "broadcast"
    REDUCESCATTER = "reducescatter"


@dsl_user_op
def warp_allreduce_sum(value, *, loc=None, ip=None):
    """
    Perform a sum all-reduce operation across all threads in a warp.
    
    Args:
        value: The value to reduce (must be numeric type)
    
    Returns:
        The sum of all values across the warp (same on all threads)
    """
    # Synchronize the warp first
    warp_sync_op(loc=loc, ip=ip)
    
    # Butterfly reduction pattern
    result = value
    for offset in [16, 8, 4, 2, 1]:
        other = shuffle_sync_op(result, offset, loc=loc, ip=ip)
        result = result + other
    
    return result


@dsl_user_op
def warp_allreduce_max(value, *, loc=None, ip=None):
    """
    Perform a max all-reduce operation across all threads in a warp.
    
    Args:
        value: The value to reduce (must be numeric type)
    
    Returns:
        The maximum value across the warp (same on all threads)
    """
    warp_sync_op(loc=loc, ip=ip)
    
    result = value
    for offset in [16, 8, 4, 2, 1]:
        other = shuffle_sync_op(result, offset, loc=loc, ip=ip)
        result = cute.max(result, other)
    
    return result


@dsl_user_op
def warp_broadcast(value, src_lane=0, *, loc=None, ip=None):
    """
    Broadcast a value from src_lane to all threads in the warp.
    
    Args:
        value: The value to broadcast
        src_lane: Source lane ID (0-31)
    
    Returns:
        The broadcasted value from src_lane
    """
    warp_sync_op(loc=loc, ip=ip)
    return shuffle_sync_op(value, src_lane, loc=loc, ip=ip)


@dsl_user_op
def warp_allgather(value, output_array, *, loc=None, ip=None):
    """
    Gather values from all threads in the warp.
    
    Args:
        value: The local value to contribute
        output_array: Array to store gathered values (size must be >= 32)
    """
    warp_sync_op(loc=loc, ip=ip)
    
    # Each thread writes its value to the appropriate position
    tid = lane_id(loc=loc, ip=ip)
    
    # Collect all values using shuffle
    for i in range(32):
        gathered_value = shuffle_sync_op(value, i, loc=loc, ip=ip)
        output_array[i] = gathered_value


@dsl_user_op
def block_allreduce_sum(value, shared_memory, *, loc=None, ip=None):
    """
    Perform a sum all-reduce operation across all threads in a block.
    
    Args:
        value: The value to reduce
        shared_memory: Shared memory array for intermediate results
    
    Returns:
        The sum of all values across the block
    """
    tid = thread_idx(loc=loc, ip=ip)
    warp_id_val = warp_id(loc=loc, ip=ip)
    lane_id_val = lane_id(loc=loc, ip=ip)
    
    # First, reduce within each warp
    warp_result = warp_allreduce_sum(value, loc=loc, ip=ip)
    
    # Store warp results in shared memory (one per warp)
    if lane_id_val == 0:
        shared_memory[warp_id_val] = warp_result
    
    # Synchronize all threads in the block
    fence_acq_rel_cta(loc=loc, ip=ip)
    
    # Reduce across warps using the first warp
    if warp_id_val == 0:
        # Load warp results (pad with zeros if needed)
        warp_data = shared_memory[lane_id_val] if lane_id_val < 32 else 0
        final_result = warp_allreduce_sum(warp_data, loc=loc, ip=ip)
        
        # Store final result back to shared memory
        if lane_id_val == 0:
            shared_memory[0] = final_result
    
    # Final synchronization
    fence_acq_rel_cta(loc=loc, ip=ip)
    
    # All threads read the final result
    return shared_memory[0]


@dsl_user_op
def cluster_broadcast(value, src_block_id=0, shared_memory=None, *, loc=None, ip=None):
    """
    Broadcast a value from one block to all blocks in a cluster.
    
    Args:
        value: The value to broadcast (only matters on src_block)
        src_block_id: The source block ID
        shared_memory: Shared memory for coordination
    
    Returns:
        The broadcasted value
    """
    block_id = cute.block_idx_x()
    
    # Store value in shared memory on source block
    if block_id == src_block_id:
        if shared_memory is not None:
            shared_memory[0] = value
    
    # Cluster-level synchronization
    fence_acq_rel_cluster(loc=loc, ip=ip)
    
    # All blocks read from the source block's shared memory
    # Note: This requires cluster shared memory or other coordination mechanism
    if shared_memory is not None:
        return shared_memory[0]
    else:
        return value


# High-level communication kernel examples
@cute.jit
def communication_benchmark_kernel(
    input_data: cute.TensorView,
    output_data: cute.TensorView,
    shared_buffer: cute.SharedMemory,
    pattern: Int32
):
    """
    Benchmark kernel for different communication patterns.
    
    Args:
        input_data: Input tensor view
        output_data: Output tensor view  
        shared_buffer: Shared memory buffer for communication
        pattern: Communication pattern ID
    """
    tid = thread_idx()
    
    # Load input value
    local_value = input_data[tid]
    
    # Apply communication pattern based on pattern ID
    if pattern == 0:  # AllReduce Sum
        result = warp_allreduce_sum(local_value)
        
    elif pattern == 1:  # AllReduce Max
        result = warp_allreduce_max(local_value)
        
    elif pattern == 2:  # Broadcast from lane 0
        result = warp_broadcast(local_value, src_lane=0)
        
    elif pattern == 3:  # Block-level AllReduce
        result = block_allreduce_sum(local_value, shared_buffer)
        
    else:  # Default: no communication
        result = local_value
    
    # Store result
    output_data[tid] = result


@cute.jit
def gradient_allreduce_kernel(
    gradients: cute.TensorView,
    reduced_gradients: cute.TensorView,
    shared_buffer: cute.SharedMemory,
    world_size: Int32
):
    """
    Example kernel for gradient all-reduce in distributed training.
    
    Args:
        gradients: Input gradients tensor
        reduced_gradients: Output reduced gradients
        shared_buffer: Shared memory for reductions
        world_size: Number of processes in the world
    """
    tid = thread_idx()
    
    # Load gradient value
    grad_value = gradients[tid]
    
    # Perform all-reduce within the block
    reduced_value = block_allreduce_sum(grad_value, shared_buffer)
    
    # Average by world size (for gradient averaging)
    averaged_grad = reduced_value / world_size
    
    # Store the averaged gradient
    reduced_gradients[tid] = averaged_grad


@cute.jit  
def parameter_broadcast_kernel(
    parameters: cute.TensorView,
    broadcasted_params: cute.TensorView,
    root_rank: Int32
):
    """
    Example kernel for parameter broadcasting from root rank.
    
    Args:
        parameters: Input parameters (only valid on root rank)
        broadcasted_params: Output broadcasted parameters
        root_rank: Rank of the root process
    """
    tid = thread_idx()
    block_id = cute.block_idx_x()
    
    # Load parameter (only meaningful on root rank)
    param_value = parameters[tid]
    
    # Broadcast within warp first
    if block_id == root_rank:
        # On root rank, broadcast from lane 0
        broadcasted_value = warp_broadcast(param_value, src_lane=0)
    else:
        # On other ranks, use dummy value
        broadcasted_value = warp_broadcast(0.0, src_lane=0)
    
    # Store result
    broadcasted_params[tid] = broadcasted_value


# Utility functions for setting up communication
def setup_communication_kernel(pattern: CommPattern, block_size: int = 256):
    """
    Helper function to set up a communication kernel with appropriate configuration.
    
    Args:
        pattern: The communication pattern to use
        block_size: Number of threads per block
    
    Returns:
        Configured kernel function
    """
    config = cute.LaunchConfig(
        block=(block_size, 1, 1),
        grid=(1, 1, 1),  # Single block for warp-level operations
        shared_memory_size=4096  # 4KB shared memory
    )
    
    if pattern == CommPattern.ALLREDUCE:
        return communication_benchmark_kernel.configure(config)
    elif pattern == CommPattern.BROADCAST:
        return parameter_broadcast_kernel.configure(config)
    elif pattern == CommPattern.ALLGATHER:
        # Configure for all-gather pattern
        return communication_benchmark_kernel.configure(config)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")


# Example usage
def example_communication_usage():
    """
    Example of how to use the custom communication primitives.
    """
    import numpy as np
    
    # Setup
    block_size = 256
    data_size = block_size
    
    # Create test data
    input_data = np.random.rand(data_size).astype(np.float32)
    output_data = np.zeros_like(input_data)
    
    # Configure kernel for all-reduce
    kernel = setup_communication_kernel(CommPattern.ALLREDUCE, block_size)
    
    # Launch kernel
    pattern_id = 0  # AllReduce Sum
    kernel(input_data, output_data, pattern_id)
    
    print(f"Input sum: {np.sum(input_data)}")
    print(f"Output (should be same on all elements): {output_data[0]}")
    print(f"All elements equal: {np.allclose(output_data, output_data[0])}")


if __name__ == "__main__":
    example_communication_usage()