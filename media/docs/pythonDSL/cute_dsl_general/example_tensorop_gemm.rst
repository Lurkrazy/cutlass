.. _example_tensorop_gemm:

Tensor Core GEMM Example
========================

The Tensor Core GEMM example demonstrates CuTeDSL's capabilities for implementing high-performance matrix multiplication using NVIDIA Ampere Tensor Cores. This example showcases advanced mixed-precision computation, complex epilogue operations, and integration with specialized hardware instructions.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/tensorop_gemm.py``

**Purpose**: Implement dense GEMM (C = A * B) using Ampere Tensor Cores while demonstrating:

- Mixed-precision computation (FP16 inputs, FP32 accumulation)
- Tensor Core MMA instruction integration
- Epilogue fusion with type conversion and scaling
- Batch dimension handling
- Shared memory buffering for epilogue operations
- Complex layout transformations for Tensor Core requirements

Key Tensor Core Concepts
------------------------

Mixed-Precision Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports sophisticated mixed-precision configurations:

.. code-block:: python

   class TensorOpGemm:
       def __init__(
           self,
           ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,     # Input precision
           c_dtype: Type[cutlass.Numeric] = cutlass.Float16,      # Output precision  
           acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,    # Accumulator precision
           cta_tiler: Tuple[int, int, int] = (128, 128, 32),
           atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
           num_stages: int = 3,
       ):

**Precision Benefits**:
- **FP16 inputs**: Reduce memory bandwidth and storage requirements
- **FP32 accumulation**: Maintain numerical accuracy during computation
- **FP16 outputs**: Minimize memory traffic for results

Tensor Core MMA Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides direct access to Tensor Core instructions:

.. code-block:: python

   # Define Tensor Core MMA operation
   op = cute.nvgpu.MmaOp(
       ab_dtype,     # Input A/B data type
       c_dtype,      # Output C data type  
       acc_dtype,    # Accumulator data type
       cute.make_shape(16, 8, 16),  # MMA instruction shape (M, N, K)
   )
   
   # Create tiled MMA with atom layout configuration
   tiled_mma = cute.make_tiled_mma(
       op,
       atom_layout,                                    # Thread organization
       permutation_mnk=(None, None, permutation_K),   # Data reordering
   )

**Tensor Core Features**:
- ``cute.nvgpu.MmaOp``: Wrapper for Tensor Core instructions
- Automatic shape matching: ``(16, 8, 16)`` for FP16 operations
- Permutation support for data layout transformations

Atom Layout Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Tensor Core operations require specific thread and data organization:

.. code-block:: python

   # Atom layout defines MMA thread arrangement
   atom_layout = cute.make_layout(
       atom_layout_mnk,               # (2, 2, 1) - threads per MMA
       stride=(atom_layout_mnk[1], 1, 0)  # Thread stride pattern
   )
   
   # Permutation for K-dimension data reordering
   permutation_K = cute.make_layout(
       (cute.size(atom_layout, mode=[2]), self._bK // cute.size(atom_layout, mode=[2])),
       stride=(self._bK // cute.size(atom_layout, mode=[2]), 1)
   )

**Layout Requirements**:
- Atom layouts must match Tensor Core instruction requirements
- K-dimension permutations optimize data access patterns
- Thread arrangements must align with warp-level operations

Batch Processing Support
~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports batch operations through layout design:

.. code-block:: python

   # Batch coordinate handling in kernel launch
   def __call__(self, mA, mB, mC, ...):
       # Grid computation includes batch dimension
       grid_dim = (
           *cute.ceil_div(mC.shape[:2], (self._bM, self._bN)),  # M, N dimensions
           mC.shape[2],  # Batch dimension
       )
       
       self.kernel(...).launch(
           grid=grid_dim,    # (grid_m, grid_n, batch_size)
           block=[cute.size(atom_layout), 1, 1],
           smem=smem_size,
           stream=stream,
       )

Epilogue Operations and Type Conversion
---------------------------------------

Shared Memory Epilogue Buffering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports complex epilogue patterns with shared memory buffering:

.. code-block:: python

   # Shared memory allocation for epilogue
   sC_layout = cute.make_layout(
       (self._bM, self._bN),
       stride=(1, self._bM + padding_c)  # Add padding for bank conflicts
   )
   sC = cute.make_tensor(
       cute.recast_ptr(sA.iterator, dtype=self.c_dtype), 
       sC_layout
   )

**Epilogue Benefits**:
- Coalesced global memory writes
- Opportunity for data type conversion
- Reduced memory bank conflicts

Mixed-Precision Epilogue
~~~~~~~~~~~~~~~~~~~~~~~

Complex type conversion and scaling in the epilogue:

.. code-block:: python

   @cute.kernel
   def kernel(self, mA, mB, mC, ...):
       # ... mainloop computation in FP32 ...
       
       # Type conversion and scaling in epilogue
       tCrC.store(
           cutlass.cast(self.c_dtype, epilogue_op(tCrC.load()))
       )
       
       # Copy to shared memory for coalesced write
       cute.copy(tiled_copy_C_epilogue, tCrC, tCsC_epilogue)
       
       # Final write to global memory
       cute.copy(tiled_copy_C, tCsC_epilogue, tCgC_epilogue)

**Type Conversion**:
- ``cutlass.cast()``: Explicit type conversion between precisions
- Epilogue operations applied before type conversion
- Automatic precision handling in copy operations

Advanced Memory Access Patterns
-------------------------------

Tensor Core Memory Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized layouts for Tensor Core efficiency:

.. code-block:: python

   # Tiled copy configuration for Tensor Core data
   tiled_copy_A = cute.make_tiled_copy(
       cute.make_copy_atom(cute.nvgpu.copy.CopyAtomGlobalToShared(), ab_dtype),
       cute.make_layout((32, 8), stride=(8, 1)),      # Thread layout
       cute.make_layout((1, 8), stride=(8, 1)),       # Value layout  
   )
   
   # Ensure layouts align with Tensor Core requirements
   assert cute.size(tiled_copy_A.get_layout_S()) == self._bM * self._bK

**Layout Constraints**:
- Thread layouts must enable coalesced global memory access
- Value layouts should match Tensor Core input requirements
- Shared memory layouts must support conflict-free access

Asynchronous Copy Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration with asynchronous memory operations:

.. code-block:: python

   # Prefetch data for Tensor Core consumption
   for k_tile in range(min(k_tile_count, k_pipe_max - 1)):
       cute.copy(
           tiled_copy_A,
           tAgA[None, None, None, k_tile],
           tAsA[None, None, None, k_tile],
           pred=tApA if k_tile > 0 else tApA_residue_k,
       )
       cute.arch.cp_async_commit_group()
   
   # Wait for data before Tensor Core operations
   cute.arch.cp_async_wait_group(k_pipe_max - 2)
   cute.arch.barrier()

Register Management for Tensor Cores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized fragment management for mixed-precision:

.. code-block:: python

   # Create fragments with correct data types
   tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])  # FP16
   tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])  # FP16  
   tCrC = tiled_mma.make_fragment_C(tCgC)                       # FP32
   
   # Initialize accumulator (must be FP32 for precision)
   tCrC.fill(cutlass.cast(acc_dtype, 0.0))
   
   # Tensor Core computation maintains precision
   cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], 
            tCrB[None, None, k_block], tCrC)

Compilation and Execution Patterns
----------------------------------

Type-Aware Compilation
~~~~~~~~~~~~~~~~~~~~~

CuTeDSL handles mixed-precision compilation automatically:

.. code-block:: python

   # Compilation with type specialization
   gemm = cute.compile(
       tensor_op_gemm,
       a_tensor,        # FP16 tensor
       b_tensor,        # FP16 tensor  
       c_tensor,        # FP16 tensor (will be converted from FP32 accumulator)
       stream=current_stream,
   )

**Compilation Features**:
- Automatic type inference and validation
- Optimized code generation for specific precision combinations
- Hardware instruction selection based on data types

Workspace Management
~~~~~~~~~~~~~~~~~~~

Efficient memory management for batch operations:

.. code-block:: python

   def generate_tensors():
       # Create workspace tensors with proper alignment
       a_workspace = create_tensor(M, K, L, ab_dtype, a_major == "m")
       b_workspace = create_tensor(N, K, L, ab_dtype, b_major == "n")  
       c_workspace = create_tensor(M, N, L, c_dtype, c_major == "m")
       
       return testing.JitArguments(
           from_dlpack(a_workspace).mark_layout_dynamic(),
           from_dlpack(b_workspace).mark_layout_dynamic(),
           from_dlpack(c_workspace).mark_layout_dynamic(),
           current_stream,
       )

Performance Optimization Techniques
----------------------------------

Threadblock Rasterization
~~~~~~~~~~~~~~~~~~~~~~~~~

Optimized grid traversal for cache locality:

.. code-block:: python

   # Grid dimensions with batch consideration
   grid_dim = (
       *cute.ceil_div(mC.shape[:2], (self._bM, self._bN)),
       mC.shape[2],  # Batch dimension processed independently
   )

**Rasterization Benefits**:
- Improved data reuse between adjacent threadblocks
- Better cache locality for large matrices
- Reduced memory traffic through spatial locality

Memory Bandwidth Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strategic layout design for maximum throughput:

.. code-block:: python

   # Padding calculation to avoid bank conflicts
   padding_a = 8 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
   padding_b = 8 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
   padding_c = 8  # Always pad epilogue buffer
   
   # Vectorized access patterns
   num_vectorized_bits = 128  # 128-bit vectorized access
   vector_size = num_vectorized_bits // ab_dtype.width

**Optimization Strategies**:
- Bank conflict avoidance through strategic padding
- Vectorized memory operations for maximum bandwidth
- Alignment-aware layout construction

Best Practices for Tensor Core Programming
------------------------------------------

1. **Precision Strategy**: Use FP16 for memory-bound operations, FP32 for compute-bound accumulation

2. **Layout Alignment**: Ensure all layouts satisfy Tensor Core alignment requirements

3. **Epilogue Design**: Buffer through shared memory for coalesced global writes

4. **Type Safety**: Use explicit type conversion for mixed-precision operations

5. **Pipeline Depth**: Maintain sufficient pipeline depth to hide Tensor Core latency

6. **Memory Optimization**: Add padding to shared memory layouts for bank conflict avoidance

7. **Batch Processing**: Design grid dimensions to handle batch operations efficiently

8. **Validation**: Test with multiple precision combinations to ensure correctness

This example demonstrates how CuTeDSL enables developers to leverage the full performance potential of Tensor Cores while maintaining code clarity and correctness through its type-safe, high-level programming model.