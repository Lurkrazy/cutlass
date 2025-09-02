.. _example_smem_allocator:

Shared Memory Allocator Example
===============================

The shared memory allocator example demonstrates CuTeDSL's sophisticated shared memory management capabilities. This example showcases how to define custom data structures, manage memory alignment, and efficiently allocate shared memory resources for GPU kernels.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/smem_allocator.py``

**Purpose**: Demonstrate advanced shared memory allocation patterns while showing:

- Custom struct definition with ``@cute.struct`` decorator
- Mixed alignment requirements (natural and strict alignment)
- Dynamic shared memory allocation with ``SmemAllocator``
- Nested struct support with alignment control
- Raw memory allocation for flexible data structures
- Tensor allocation directly in shared memory

Key Shared Memory Concepts
--------------------------

Struct Definition with Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides ``@cute.struct`` for defining custom data structures with precise alignment control:

.. code-block:: python

   @cute.struct
   class complex:
       real: cutlass.Float32
       imag: cutlass.Float32
   
   @cute.struct
   class SharedStorage:
       # Natural alignment - follows C++ alignment rules
       a: cute.struct.MemRange[cutlass.Float32, 32]  # Array of 32 floats
       b: cutlass.Int64                              # 64-bit integer  
       c: complex                                    # Nested struct
       
       # Strict alignment - explicit alignment requirements
       x: cute.struct.Align[
           cute.struct.MemRange[cutlass.Float32, 32],
           128,  # Force 128-byte alignment
       ]
       y: cute.struct.Align[cutlass.Int32, 8]        # 8-byte aligned int
       z: cute.struct.Align[complex, 16]             # 16-byte aligned complex

**Struct Features**:
- ``@cute.struct``: Decorator for GPU-compatible struct definition
- ``MemRange[Type, Size]``: Fixed-size array within struct
- ``Align[Type, Alignment]``: Explicit alignment specification
- Nested struct support with automatic alignment calculation

SmemAllocator Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic shared memory allocation through the allocator interface:

.. code-block:: python

   @cute.kernel
   def kernel(const_a, dst_a, const_b, dst_b, const_c, dst_c):
       # Initialize allocator - base pointer starts at shared memory beginning
       allocator = cutlass.utils.SmemAllocator()
       
       # Allocate struct in shared memory
       struct_in_smem = allocator.allocate(SharedStorage)
       
       # Access struct members
       struct_in_smem.a[0] = const_a
       struct_in_smem.b = const_b
       struct_in_smem.c.real = const_c

**Allocator Features**:
- ``allocator.allocate(Type)``: Type-safe allocation with automatic alignment
- Base pointer alignment guaranteed at 1024 bytes
- Sequential allocation with automatic address calculation
- No explicit deallocation required (scope-based management)

Memory Range and Array Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with arrays and memory ranges within structs:

.. code-block:: python

   # Array access within struct
   for i in cutlass.range(32, unroll=1):
       struct_in_smem.a[i] = const_a + cutlass.cast(cutlass.Float32, i)
   
   # Direct memory range manipulation
   array_ptr = struct_in_smem.a.data()  # Get pointer to array data
   
   # Copy array contents to global memory
   for i in cutlass.range(32, unroll=1):
       dst_a[i] = struct_in_smem.a[i]

**Array Operations**:
- Index-based access with bounds checking
- ``.data()`` method for raw pointer access
- Automatic loop unrolling with ``cutlass.range(n, unroll=1)``

Advanced Allocation Patterns
----------------------------

Raw Memory Block Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct memory allocation for custom data structures:

.. code-block:: python

   # Raw memory allocation with custom alignment
   raw_mem_ptr = allocator.allocate_raw(
       num_bytes=1024,    # Total bytes to allocate
       alignment=64,      # Required alignment in bytes
   )
   
   # Cast to specific pointer type
   float_ptr = cute.recast_ptr(raw_mem_ptr, cutlass.Float32)
   
   # Use as array
   for i in cutlass.range(256, unroll=1):  # 1024 bytes / 4 bytes per float
       float_ptr[i] = cutlass.cast(cutlass.Float32, i)

**Raw Allocation Benefits**:
- Maximum flexibility for custom data layouts
- Precise control over alignment requirements
- Efficient memory usage for specialized data structures

Array Allocation with Automatic Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplified array allocation with automatic type-based alignment:

.. code-block:: python

   # Allocate typed array in shared memory
   array_in_smem = allocator.allocate_array(
       element_type=cutlass.Float32,
       num_elements=128,
       alignment=16,  # Optional: override default alignment
   )
   
   # Direct array operations
   for i in cutlass.range(128, unroll=1):
       array_in_smem[i] = const_a * cutlass.cast(cutlass.Float32, i)

**Array Features**:
- Type-safe element access
- Automatic alignment based on element type
- Bounds checking in debug builds

Tensor Allocation in Shared Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct tensor creation in shared memory with layout specification:

.. code-block:: python

   # Define tensor layout
   tensor_layout = cute.make_layout(
       (16, 8),           # Shape: 16x8
       stride=(1, 16),    # Column-major layout
   )
   
   # Allocate tensor directly in shared memory
   tensor_in_smem = allocator.allocate_tensor(
       element_type=cutlass.Float32,
       layout=tensor_layout,
       alignment=128,      # Ensure proper alignment for vectorized access
   )
   
   # Tensor operations
   for i in cutlass.range(cute.size(tensor_in_smem), unroll=1):
       tensor_in_smem[i] = const_a

**Tensor Allocation Benefits**:
- Direct layout specification for optimal access patterns
- Integration with CuTe tensor operations
- Automatic stride calculation and validation

Memory Management Patterns
--------------------------

Alignment Strategy
~~~~~~~~~~~~~~~~~

Strategic alignment for optimal memory access:

.. code-block:: python

   # Bank conflict avoidance for shared memory
   @cute.struct
   class PaddedData:
       # Pad arrays to avoid bank conflicts
       data: cute.struct.Align[
           cute.struct.MemRange[cutlass.Float32, 128 + 4],  # Add padding
           128,  # High alignment for vectorized access
       ]
   
   # Vectorized access alignment
   vectorized_array = allocator.allocate_array(
       element_type=cutlass.Float32,
       num_elements=256,
       alignment=128,  # Enable 128-bit vectorized loads/stores
   )

**Alignment Guidelines**:
- Use 128-byte alignment for vectorized operations
- Add padding to avoid shared memory bank conflicts
- Align to cache line boundaries for optimal performance

Shared Memory Size Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic shared memory size computation:

.. code-block:: python

   # Calculate total shared memory requirement
   def calculate_smem_size():
       struct_size = cute.size_in_bytes(SharedStorage)
       array_size = 128 * cutlass.Float32.size  # 128 floats
       tensor_size = cute.size_in_bytes(cutlass.Float32, tensor_layout)
       raw_size = 1024  # Raw allocation
       
       total_size = struct_size + array_size + tensor_size + raw_size
       return total_size
   
   # Launch kernel with computed shared memory size
   kernel(...).launch(
       grid=[1, 1, 1],
       block=[256, 1, 1],
       smem=calculate_smem_size(),  # Dynamic shared memory
   )

**Size Calculation**:
- ``cute.size_in_bytes(Type)`` for automatic size computation
- Include padding and alignment overhead
- Validate against hardware shared memory limits

Memory Access Patterns
----------------------

Coalesced Access Through Structs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Designing structs for optimal memory access:

.. code-block:: python

   @cute.struct
   class CoalescedData:
       # Arrange data for optimal thread access patterns
       per_thread_data: cute.struct.MemRange[cutlass.Float32, 4]  # Vectorizable
       shared_data: cute.struct.Align[cutlass.Float32, 128]       # High alignment
   
   # Access pattern for coalesced reads
   tidx, _, _ = cute.arch.thread_idx()
   data_per_thread = struct_in_smem.per_thread_data
   for i in cutlass.range(4, unroll=1):
       data_per_thread[i] = global_data[tidx * 4 + i]

**Access Optimization**:
- Arrange struct fields for coalesced access
- Use vectorizable data sizes (multiples of 4/8/16 elements)
- Align frequently accessed data to cache boundaries

Shared Memory Banking Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strategic data layout to avoid bank conflicts:

.. code-block:: python

   # Bank conflict avoidance through padding
   padded_size = original_size + (original_size % 32)  # Add padding
   
   array_with_padding = allocator.allocate_array(
       element_type=cutlass.Float32,
       num_elements=padded_size,
       alignment=128,
   )
   
   # Access with stride to avoid conflicts
   stride = 33  # Prime number to distribute access
   for i in cutlass.range(num_threads, unroll=1):
       index = (i * stride) % padded_size
       local_data = array_with_padding[index]

**Banking Guidelines**:
- Add padding to break regular access patterns
- Use prime strides for irregular access
- Consider warp-level access patterns in layout design

Best Practices for Shared Memory Management
-------------------------------------------

1. **Struct Design**: Use natural alignment for most cases, strict alignment only when necessary

2. **Memory Efficiency**: Pack related data into structs to minimize allocation overhead

3. **Alignment Strategy**: Use high alignment (128 bytes) for vectorized operations

4. **Bank Conflict Avoidance**: Add strategic padding to break regular access patterns

5. **Size Planning**: Calculate shared memory requirements early and validate against hardware limits

6. **Access Patterns**: Design data layouts to match thread access patterns for coalescing

7. **Type Safety**: Leverage CuTeDSL's type system to prevent memory access errors

8. **Debug Validation**: Use bounds checking and alignment validation in debug builds

Debugging and Validation
------------------------

Memory Layout Debugging
~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides utilities for debugging memory layouts:

.. code-block:: python

   # Print struct layout information at compile time
   print(f"SharedStorage size: {cute.size_in_bytes(SharedStorage)} bytes")
   print(f"SharedStorage alignment: {cute.alignment_of(SharedStorage)} bytes")
   
   # Validate alignment requirements
   assert cute.alignment_of(SharedStorage) <= 1024, "Exceeds base alignment"

**Debugging Features**:
- Compile-time size and alignment calculation
- Layout validation and assertion support
- Memory access pattern analysis

This example demonstrates how CuTeDSL enables sophisticated shared memory management while maintaining type safety and performance optimization opportunities through its comprehensive allocation and struct definition capabilities.