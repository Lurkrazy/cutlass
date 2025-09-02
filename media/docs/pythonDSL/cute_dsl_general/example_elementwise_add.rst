.. _example_elementwise_add:

Elementwise Addition Example
===========================

The elementwise addition example demonstrates fundamental CuTeDSL concepts including tensor partitioning, memory operations, and predication. This example serves as an excellent introduction to the core CuTeDSL syntax patterns.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/elementwise_add.py``

**Purpose**: Perform elementwise addition of two tensors (C = A + B) using CuTeDSL while demonstrating:

- Basic tensor operations and memory management
- Thread-value (TV) layouts for canonical partitioning patterns  
- Global memory to register memory transfers
- Predication for out-of-bounds protection
- Kernel launch patterns

Key CuTeDSL Concepts Demonstrated
--------------------------------

Function Decorators
~~~~~~~~~~~~~~~~~~~

CuTeDSL provides two primary decorators for code generation:

.. code-block:: python

   @cute.jit
   def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
       # Host-side JIT-compiled function
       pass

   @cute.kernel  
   def elementwise_add_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):
       # GPU kernel function
       pass

- ``@cute.jit``: Declares JIT-compiled functions that can be invoked from Python
- ``@cute.kernel``: Defines GPU kernel functions that run on the device
- Functions decorated with ``@cute.jit`` can call functions decorated with ``@cute.kernel``

Tensor Creation and Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL uses layouts to define how tensors map logical coordinates to memory addresses:

.. code-block:: python

   # Create thread and value layouts for memory access patterns
   thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))  # Thread layout
   val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))  # Value layout
   
   # Combine thread and value layouts into a TV (thread-value) layout
   tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

**Thread Layout**: Maps thread IDs to 2D coordinates within a tile
**Value Layout**: Defines how each thread accesses multiple values for vectorization
**TV Layout**: Combines thread and value layouts for efficient memory access

Tensor Partitioning with Zipped Divide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``zipped_divide`` operation tiles tensors into manageable blocks:

.. code-block:: python

   # Partition input tensors into tiles
   gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
   gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
   gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))

``zipped_divide`` transforms a tensor into a hierarchical structure:
- Mode 0: ``(TileM, TileN)`` - The tile dimensions 
- Mode 1: ``(RestM, RestN)`` - The remaining dimensions for multiple tiles

Coordinate Tensors and Predication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinate tensors track logical positions for bounds checking:

.. code-block:: python

   # Create identity tensor for coordinate tracking
   idC = cute.make_identity_tensor(mC.shape)
   cC = cute.zipped_divide(idC, tiler=tiler_mn)
   
   # Generate predicate masks for out-of-bounds protection
   for i in range(0, cute.size(frgPred), 1):
       val = cute.elem_less(thrCrd[i], shape)
       frgPred[i] = val

Predication prevents out-of-bounds memory accesses when tensor dimensions are not multiples of tile sizes.

Memory Copy Operations
~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides efficient memory copy primitives:

.. code-block:: python

   # Define copy atoms for memory transfers
   copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
   copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
   
   # Create tiled copy operations
   tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
   
   # Copy from global memory to register memory with predication
   cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)

Kernel Launch Syntax
~~~~~~~~~~~~~~~~~~~~

Kernels are launched with explicit grid and block dimensions:

.. code-block:: python

   elementwise_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
       grid=[cute.size(gC, mode=[1]), 1, 1],    # Grid dimensions
       block=[cute.size(tv_layout, mode=[0]), 1, 1],  # Block dimensions
   )

Register Operations and Fragment Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL manages register memory through fragments:

.. code-block:: python

   # Allocate register fragments
   frgA = cute.make_fragment_like(thrA)
   frgB = cute.make_fragment_like(thrB) 
   frgC = cute.make_fragment_like(thrC)
   
   # Load data from fragments, perform computation, store results
   result = frgA.load() + frgB.load()
   frgC.store(result)

Thread and Block Index Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access CUDA thread and block indices using architecture intrinsics:

.. code-block:: python

   tidx, _, _ = cute.arch.thread_idx()  # Thread index within block
   bidx, _, _ = cute.arch.block_idx()   # Block index within grid

Syntax Patterns and Best Practices
----------------------------------

1. **Layout Definition Order**: Use ``order=(1, 0)`` to specify row-major layouts for optimal memory coalescing

2. **Fragment Management**: Always use ``make_fragment_like()`` to create fragments that match tensor partitioning

3. **Predication**: Apply predicates to all memory operations that might access out-of-bounds memory

4. **Type Consistency**: Ensure copy atoms match the element types of source and destination tensors

5. **Vectorization**: Use appropriate vector sizes (e.g., 4 elements for 128-bit copies with 32-bit elements)

Code Structure Walkthrough
--------------------------

1. **Setup Phase** (``@cute.jit`` function):
   - Define thread and value layouts for memory access patterns
   - Create TV layout combining thread mapping and vectorization
   - Partition input tensors using ``zipped_divide``
   - Create coordinate tensors for bounds checking

2. **Kernel Execution** (``@cute.kernel`` function):
   - Extract thread and block indices
   - Slice tensors for current thread block
   - Create copy operations and partition tensors per thread
   - Allocate register fragments for computation
   - Generate predicate masks for bounds checking
   - Perform memory copies with predication
   - Execute elementwise computation in registers
   - Store results back to global memory

3. **Launch Configuration**:
   - Set grid dimensions based on tensor tiling
   - Configure block dimensions from TV layout
   - Launch kernel with computed parameters

This example establishes the foundational patterns used throughout CuTeDSL for memory management, tensor operations, and kernel organization.