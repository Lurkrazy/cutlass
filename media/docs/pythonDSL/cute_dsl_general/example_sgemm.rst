.. _example_sgemm:

SIMT GEMM Example
=================

The SIMT GEMM example demonstrates advanced CuTeDSL patterns for implementing high-performance matrix multiplication using scalar floating-point units (SIMT). This example showcases sophisticated memory management, multi-stage pipelines, and complex layout transformations.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/sgemm.py``

**Purpose**: Implement dense matrix multiplication (C = A * B) using SIMT cores while demonstrating:

- Object-oriented kernel design with classes
- Multi-stage shared memory pipelines 
- Asynchronous memory copy operations
- Complex predication for irregular tile shapes
- Register and shared memory buffer management
- Advanced layout composition and transformation

Key CuTeDSL Advanced Concepts
----------------------------

Object-Oriented Kernel Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports organizing kernels within classes for better modularity:

.. code-block:: python

   class SGemm:
       def __init__(self, cta_tiler=(128, 128, 8), num_stages=3, num_threads=256):
           self._cta_tiler = cta_tiler
           self._num_stages = num_stages
           self._num_threads = num_threads
       
       @cute.jit
       def __call__(self, mA, mB, mC, epilogue_op=lambda x: x, stream=...):
           # Host-side setup and kernel launch
           pass
       
       @cute.kernel  
       def kernel(self, mA, mB, mC, sA_layout, sB_layout, ...):
           # Device-side kernel implementation
           pass

**Design Benefits**:
- Encapsulation of kernel parameters and configuration
- Reusable kernel implementations with different configurations
- Clear separation between setup logic and computation

Shared Memory Layout Design
~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced shared memory layout construction with padding for bank conflict avoidance:

.. code-block:: python

   # Add padding when input tensors are k-major to reduce bank conflicts
   padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
   padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
   
   # Shared memory layouts with multi-stage buffering
   sA_layout = cute.make_layout(
       (self._bM, self._bK, self._num_stages),
       stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
   )
   sB_layout = cute.make_layout(
       (self._bN, self._bK, self._num_stages),
       stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
   )

**Layout Features**:
- Three-mode layouts: ``(tile_m/n, tile_k, stage)``
- Automatic padding calculation based on input tensor layout
- Multi-stage buffering for pipeline overlap
- Bank conflict avoidance through strategic padding

Asynchronous Memory Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides primitives for asynchronous memory operations:

.. code-block:: python

   # Create asynchronous copy atoms
   atom_async_copy_A = cute.make_copy_atom(
       cute.nvgpu.cpasync.CopyG2SOp(),    # Global-to-shared async copy
       mA.element_type,
       num_bits_per_copy=mA.element_type.width,
   )
   
   # Vectorized copies for column-major tensors
   if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
       num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
       atom_async_copy_A = cute.make_copy_atom(
           cute.nvgpu.cpasync.CopyG2SOp(),
           mA.element_type,
           num_bits_per_copy=mA.element_type.width * num_vectorized,
       )

**Pipeline Operations**:
- ``cute.nvgpu.cpasync.CopyG2SOp()``: Asynchronous global-to-shared memory copy
- ``cute.arch.cp_async_commit_group()``: Commit copy operations to pipeline
- ``cute.arch.cp_async_wait_group(n)``: Wait for pipeline to have ≤n pending operations
- ``cute.arch.barrier()``: Synchronize threads within a block

Advanced Tensor Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex tensor slicing with projection and offset operations:

.. code-block:: python

   # Local tile extraction with projection
   gA = cute.local_tile(
       mA, 
       tiler=self._cta_tiler, 
       coord=tiler_coord, 
       proj=(1, None, 1)    # Project modes: keep M, slice N, keep K
   )
   gB = cute.local_tile(
       mB,
       tiler=self._cta_tiler,
       coord=tiler_coord,
       proj=(None, 1, 1)    # Project modes: slice M, keep N, keep K  
   )
   
   # Domain offset for irregular K handling
   residue_k = mA.shape[1] - cutlass.Int32(self._bK) * gA.shape[2]
   gA = cute.domain_offset((0, residue_k, 0), gA)
   gB = cute.domain_offset((0, residue_k, 0), gB)

**Partitioning Concepts**:
- ``local_tile()``: Extract tiles for thread blocks with projections
- ``proj`` parameter controls which modes are sliced vs. preserved
- ``domain_offset()``: Shifts tensor domain for irregular tile handling

Sophisticated Predication
~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-level predication for different pipeline stages:

.. code-block:: python

   # Predicate tensors for mainloop (M/N bounds only)
   tApA = cute.make_fragment(
       cute.make_layout(
           (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
           stride=(cute.size(tAsA, mode=[1]), 1, 0),
       ),
       cutlass.Boolean,
   )
   
   # Predicate tensors for residue K handling (M/N/K bounds)
   tApA_residue_k = cute.make_fragment(
       cute.make_layout(
           (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
           stride=(
               cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
               cute.size(tAsA, mode=[2]),
               1,
           ),
       ),
       cutlass.Boolean,
   )

**Predication Levels**:
- **Mainloop predicates**: Only check M/N bounds (K assumed regular)
- **Residue predicates**: Check M/N/K bounds for irregular first tile
- **Epilogue predicates**: Check output bounds for final write

MMA Layout Configuration
~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports complex MMA (Matrix Multiply-Accumulate) layouts:

.. code-block:: python

   # MMA atom and tiled layout configuration
   atoms_layout = cute.make_layout(
       (self._num_threads // 16, 16, 1), 
       stride=(16, 1, 0)
   )
   
   # Handle different output majorness
   if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
       atoms_layout = cute.make_layout(
           (16, self._num_threads // 16, 1), 
           stride=(1, 16, 0)
       )
   
   # Permutation tilers for register reordering
   permutation_tiler_M = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
   permutation_tiler_N = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
   
   # Create tiled MMA with permutations
   tiled_mma = cute.make_tiled_mma(
       cute.nvgpu.MmaUniversalOp(cutlass.Float32),
       atoms_layout,
       permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
   )

**MMA Configuration**:
- ``atoms_layout``: Thread organization for MMA operations
- ``permutation_tiler``: Controls register access patterns
- ``MmaUniversalOp``: Flexible MMA operation for different data types

Pipeline Implementation Patterns
--------------------------------

Multi-Stage Shared Memory Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel implements a sophisticated 3-stage pipeline:

.. code-block:: python

   # Pipeline stage variables
   gmem_pipe_read = cutlass.Int32(0)     # Global memory read stage
   smem_pipe_read = cutlass.Int32(0)     # Shared memory read stage  
   smem_pipe_write = cutlass.Int32(k_pipe_max - 1)  # Shared memory write stage
   
   # Prefetch: Start async loads for multiple stages
   for k_tile in range(1, k_pipe_max - 1):
       if k_tile < k_tile_count:
           cute.copy(tiled_copy_A, tAgA[None, None, None, gmem_pipe_read], 
                    tAsA[None, None, None, k_tile], pred=tApA)
           cute.copy(tiled_copy_B, tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile], pred=tBpB)
       cute.arch.cp_async_commit_group()

**Pipeline Stages**:
1. **Prefetch**: Fill pipeline with initial data
2. **Mainloop**: Overlapped computation and memory transfer
3. **Epilogue**: Final computation and result storage

Register Pipeline for Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overlapped register operations within the mainloop:

.. code-block:: python

   for _ in range(k_tile_count):
       for k_block in range(k_block_max, unroll_full=True):
           # Wait for data availability
           if k_block == k_block_max - 1:
               cute.arch.cp_async_wait_group(k_pipe_max - 2)
               cute.arch.barrier()
           
           # Prefetch next data while computing current
           k_block_next = (k_block + 1) % k_block_max
           cute.autovec_copy(tCsA_p[None, None, k_block_next], 
                           tCrA[None, None, k_block_next])
           
           # Perform computation on current data
           cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], 
                    tCrB[None, None, k_block], tCrC)

**Register Pipeline Benefits**:
- Eliminates false dependencies between consecutive operations
- Enables better instruction-level parallelism
- Overlaps memory transfers with computation

Advanced Syntax Patterns
------------------------

Conditional Compilation
~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports compile-time conditionals:

.. code-block:: python

   if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
       # Compile-time branch for column-major tensors
       num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
       # ... vectorized copy setup
   
   # The compiler eliminates unused branches entirely

Fragment and Layout Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced fragment operations for register management:

.. code-block:: python

   # Create fragments matching MMA requirements
   tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
   tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0]) 
   tCrC = tiled_mma.make_fragment_C(tCgC)
   
   # Initialize accumulator
   tCrC.fill(0.0)
   
   # Epilogue operations
   tCrC.store(epilogue_op(tCrC.load()))

**Fragment Operations**:
- ``make_fragment_A/B/C()``: Create fragments matching MMA layouts
- ``fill()``: Initialize fragment contents
- ``load()`` / ``store()``: Transfer between fragments and registers

Memory Management
~~~~~~~~~~~~~~~~

Shared memory allocation and management:

.. code-block:: python

   # Calculate shared memory requirements
   smem_size = cute.size_in_bytes(mA.element_type, sA_layout) + \
               cute.size_in_bytes(mB.element_type, sB_layout)
   
   # Allocate shared memory buffers
   smem = cutlass.utils.SmemAllocator()
   sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)  # 16-byte aligned
   sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
   
   # Launch with shared memory size
   self.kernel(...).launch(
       grid=grid_dim,
       block=[cute.size(atoms_layout), 1, 1],
       smem=smem_size,    # Specify shared memory requirement
       stream=stream,
   )

Best Practices for Complex Kernels
----------------------------------

1. **Pipeline Depth**: Use 3+ stages for effective memory/compute overlap

2. **Padding Strategy**: Add padding to shared memory layouts for bank conflict avoidance

3. **Predication Hierarchy**: Use different predicate tensors for different pipeline stages

4. **Register Reuse**: Design register pipelines to minimize register pressure

5. **Vectorization**: Leverage layout alignment information for optimal vector operations

6. **Conditional Compilation**: Use ``cutlass.const_expr()`` for compile-time optimizations

7. **Memory Alignment**: Ensure shared memory allocations respect alignment requirements

This example demonstrates how CuTeDSL enables the implementation of complex, production-quality kernels with sophisticated memory management and optimization strategies while maintaining code clarity and modularity.