.. _example_flash_attention:

Flash Attention v2 Example
==========================

The Flash Attention v2 example demonstrates the most advanced CuTeDSL patterns for implementing complex fused kernels. This example showcases online softmax fusion, memory-efficient attention computation, and sophisticated pipeline management for state-of-the-art attention mechanisms.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/flash_attention_v2.py``

**Purpose**: Implement Flash Attention v2 forward pass while demonstrating:

- Online softmax computation with incremental statistics
- Complex memory pipeline orchestration
- Causal and padding mask integration
- Multi-head attention batch processing
- Advanced epilogue fusion patterns
- Warp-level primitive integration
- Numerical stability techniques for attention computation

Key Flash Attention Concepts
----------------------------

Online Softmax Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL enables implementation of the online softmax algorithm for memory efficiency:

.. code-block:: python

   @cute.kernel
   def flash_attention_kernel(self, ...):
       # Initialize statistics for online softmax
       max_val = cute.make_fragment_like(
           cute.make_layout(self.m_block_size),
           cutlass.cast(acc_dtype, float('-inf'))
       )
       sum_val = cute.make_fragment_like(
           cute.make_layout(self.m_block_size), 
           cutlass.cast(acc_dtype, 0.0)
       )
       
       # Incremental softmax update per iteration
       for sk_tile_idx in range(sk_tile_count):
           # Compute S = Q @ K^T
           cute.gemm(tiled_mma, S_frag, Q_frag, K_frag, S_frag)
           
           # Update max and sum statistics
           new_max = cute.max(max_val, cute.reduce_max(S_frag, axis=1))
           
           # Rescale previous values and accumulate
           scale_factor = cute.exp(max_val - new_max)
           sum_val = sum_val * scale_factor + cute.reduce_sum(
               cute.exp(S_frag - new_max), axis=1
           )
           max_val = new_max

**Online Softmax Benefits**:
- Constant memory usage regardless of sequence length
- Numerical stability through incremental max tracking
- Elimination of separate softmax pass

Causal and Padding Mask Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sophisticated masking patterns for different attention types:

.. code-block:: python

   # Causal mask application
   if self.is_causal:
       for m in cutlass.range(S_frag.shape[0], unroll=1):
           for n in cutlass.range(S_frag.shape[1], unroll=1):
               q_idx = m_tile_idx * self.m_block_size + m
               k_idx = sk_tile_idx * self.n_block_size + n
               
               if q_idx < k_idx:  # Upper triangular mask
                   S_frag[m, n] = cutlass.cast(acc_dtype, float('-inf'))
   
   # Padding mask for variable sequence lengths
   if has_padding_mask:
       for m in cutlass.range(S_frag.shape[0], unroll=1):
           for n in cutlass.range(S_frag.shape[1], unroll=1):
               k_idx = sk_tile_idx * self.n_block_size + n
               if k_idx >= actual_seq_len:
                   S_frag[m, n] = cutlass.cast(acc_dtype, float('-inf'))

**Masking Patterns**:
- Causal masking for autoregressive models
- Padding masking for variable sequence lengths
- Custom attention patterns through configurable masks

Multi-Head Attention Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient batch and multi-head processing:

.. code-block:: python

   class FlashAttentionForwardAmpere:
       def __init__(
           self,
           head_dim: int = 128,
           m_block_size: int = 128,
           n_block_size: int = 128,
           num_threads: int = 128,
       ):
           # Validate configuration constraints
           assert head_dim % 8 == 0, "Head dimension must be multiple of 8"
           assert m_block_size % 32 == 0, "Block size must be multiple of 32"
           assert (m_block_size * 2) % num_threads == 0, "Thread layout constraint"
   
       @cute.jit
       def __call__(self, Q, K, V, O, softmax_scale=1.0, is_causal=False):
           # Grid calculation for batch and multi-head processing
           batch_size, seqlen_q, num_head, head_dim = Q.shape
           
           grid_dim = (
               cute.ceil_div(seqlen_q, self.m_block_size),  # Sequence chunks
               num_head,                                     # Head parallelism
               batch_size,                                   # Batch parallelism
           )

**Multi-Head Features**:
- Parallel processing across attention heads
- Batch dimension handling with optimal grid layout
- Memory-efficient head dimension constraints

Advanced Pipeline Management
---------------------------

Warp-Level Primitive Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides access to warp-level operations for efficiency:

.. code-block:: python

   # Warp-level reduction operations
   from cutlass.cute.nvgpu import warp
   
   # Efficient softmax computation using warp primitives
   warp_max = warp.reduce_max(thread_local_max)
   warp_sum = warp.reduce_sum(thread_local_sum)
   
   # Broadcast results across warp
   broadcast_max = warp.broadcast(warp_max, lane_id=0)
   broadcast_sum = warp.broadcast(warp_sum, lane_id=0)

**Warp Operations**:
- ``warp.reduce_max/sum()``: Efficient warp-level reductions
- ``warp.broadcast()``: Distribute values across warp lanes
- ``warp.ballot()``: Predicate voting across warp

CpAsync Integration for Memory Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced asynchronous memory operations for attention computation:

.. code-block:: python

   # CpAsync operations for efficient memory transfers
   from cutlass.cute.nvgpu import cpasync
   
   # Asynchronous loads for Q, K, V matrices
   cute.copy(
       cute.make_copy_atom(cpasync.CopyG2SOp(), q_dtype),
       global_Q_tile,
       shared_Q_buffer,
       pred=q_predicate_mask
   )
   
   # Commit copy group and manage pipeline
   cute.arch.cp_async_commit_group()
   
   # Wait for specific pipeline stage
   cute.arch.cp_async_wait_group(pipeline_depth - 2)
   cute.arch.barrier()

**Pipeline Stages**:
- Overlapped Q/K/V loading with computation
- Multi-stage buffering for continuous data flow
- Precise synchronization control

Numerical Stability Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of numerically stable attention computation:

.. code-block:: python

   # Stable softmax with max subtraction
   def stable_softmax(S_frag, max_val):
       # Subtract max for numerical stability
       S_stable = S_frag - max_val.broadcast()
       
       # Compute exponentials
       S_exp = cute.exp(S_stable)
       
       # Normalize by sum
       sum_val = cute.reduce_sum(S_exp, axis=1)
       return S_exp / sum_val.broadcast()
   
   # Incremental scaling for online computation
   def rescale_output(O_frag, old_sum, new_sum, old_max, new_max):
       correction_factor = (old_sum / new_sum) * cute.exp(old_max - new_max)
       return O_frag * correction_factor

**Stability Features**:
- Max subtraction before exponential computation
- Incremental scaling for numerical precision
- Guard against overflow in attention scores

Complex Memory Layout Management
-------------------------------

Multi-Tensor Layout Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinated layout design for optimal memory access:

.. code-block:: python

   # Q, K, V layout configuration for attention
   def setup_attention_layouts(self, Q, K, V):
       # Ensure compatible layouts for efficient GEMM operations
       q_layout = self.create_qkv_layout(Q.shape, major_dim="seqlen")
       k_layout = self.create_qkv_layout(K.shape, major_dim="head_dim") 
       v_layout = self.create_qkv_layout(V.shape, major_dim="seqlen")
       
       # Tiled layouts for thread block processing
       tiled_q = cute.zipped_divide(Q, self.q_tiler)
       tiled_k = cute.zipped_divide(K, self.k_tiler)
       tiled_v = cute.zipped_divide(V, self.v_tiler)
       
       return tiled_q, tiled_k, tiled_v

**Layout Coordination**:
- Matching layouts for efficient tensor operations
- Optimal tiling strategies for different tensor roles
- Memory access pattern optimization

Shared Memory Buffer Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex shared memory allocation for multiple tensors:

.. code-block:: python

   @cute.kernel
   def flash_attention_kernel(self, Q, K, V, O, ...):
       # Allocate shared memory buffers
       smem_allocator = cutlass.utils.SmemAllocator()
       
       # Q buffer - input queries
       sQ = smem_allocator.allocate_tensor(
           Q.element_type, 
           cute.make_layout((self.m_block_size, self.head_dim)),
           alignment=128
       )
       
       # K buffer - keys for current iteration  
       sK = smem_allocator.allocate_tensor(
           K.element_type,
           cute.make_layout((self.n_block_size, self.head_dim)),
           alignment=128
       )
       
       # V buffer - values for current iteration
       sV = smem_allocator.allocate_tensor(
           V.element_type,
           cute.make_layout((self.n_block_size, self.head_dim)), 
           alignment=128
       )

**Buffer Management**:
- Strategic allocation order for memory efficiency
- Alignment requirements for vectorized operations
- Buffer reuse across pipeline stages

Advanced Fusion Patterns
------------------------

Epilogue Integration
~~~~~~~~~~~~~~~~~~~

Integrated epilogue operations within attention computation:

.. code-block:: python

   # Fused scaling and output computation
   for v_tile_idx in range(v_tile_count):
       # Load V tile
       cute.copy(tiled_copy_V, global_V_tile, shared_V_buffer)
       
       # Compute O += P @ V (where P is softmax output)
       cute.gemm(tiled_mma, O_accumulator, P_frag, V_frag, O_accumulator)
       
       # Apply final scaling and normalization
       if v_tile_idx == v_tile_count - 1:  # Final iteration
           # Normalize by final sum
           O_normalized = O_accumulator / final_sum.broadcast()
           
           # Apply output scaling
           O_scaled = O_normalized * output_scale
           
           # Store to global memory
           cute.copy(tiled_copy_O, O_scaled, global_O_tile)

**Fusion Benefits**:
- Reduced memory traffic through intermediate elimination
- Improved numerical stability
- Higher computational intensity

Performance Optimization Patterns
---------------------------------

Block Size Tuning
~~~~~~~~~~~~~~~~~

Strategic block size selection for optimal performance:

.. code-block:: python

   # Performance-oriented block size selection
   def get_optimal_block_sizes(head_dim, seq_len, available_smem):
       if head_dim <= 64:
           m_block = 128
           n_block = 128
       elif head_dim <= 128:
           m_block = 64
           n_block = 128  
       else:
           m_block = 32
           n_block = 64
       
       # Validate shared memory requirements
       smem_required = calculate_smem_usage(m_block, n_block, head_dim)
       assert smem_required <= available_smem
       
       return m_block, n_block

**Tuning Considerations**:
- Head dimension impact on register usage
- Shared memory capacity constraints
- Thread block occupancy optimization

Memory Access Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Strategic memory access patterns for bandwidth efficiency:

.. code-block:: python

   # Vectorized loading with optimal stride patterns
   vectorized_copy = cute.make_tiled_copy(
       cute.make_copy_atom(
           cute.nvgpu.CopyUniversalOp(),
           element_type,
           num_bits_per_copy=128,  # 128-bit vectorized access
       ),
       thread_layout,
       value_layout,
   )
   
   # Bank conflict avoidance in shared memory
   padded_layout = cute.make_layout(
       (block_size, head_dim + padding),  # Add padding
       stride=(1, block_size + padding),
   )

**Access Optimization**:
- Vectorized memory operations for bandwidth
- Bank conflict avoidance through padding
- Coalesced access pattern design

Best Practices for Complex Fused Kernels
----------------------------------------

1. **Numerical Stability**: Always use online algorithms for softmax computation

2. **Memory Efficiency**: Implement tiling strategies that fit in shared memory

3. **Pipeline Design**: Overlap computation with memory transfers using async operations

4. **Mask Integration**: Fuse masking operations into computation loops

5. **Precision Management**: Use appropriate precision for different computation stages

6. **Warp Utilization**: Leverage warp-level primitives for efficient reductions

7. **Layout Coordination**: Design coordinated layouts across multiple tensors

8. **Validation**: Test with reference implementations for correctness

9. **Performance Tuning**: Profile and optimize block sizes for target hardware

10. **Code Organization**: Maintain clear separation between setup and computation logic

This example demonstrates how CuTeDSL enables the implementation of sophisticated, production-quality fused kernels that achieve both high performance and numerical stability through careful algorithm design and optimization.