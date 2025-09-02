.. _example_call_from_jit:

JIT Function Integration Example
===============================

The call_from_jit example demonstrates advanced CuTeDSL patterns for integrating JIT-compiled functions with static shape operations and custom argument protocols. This showcases how to build reusable, modular kernel components that can be composed into larger applications.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/call_from_jit.py``

**Purpose**: Demonstrate JIT function composition and integration patterns while showing:

- Custom argument protocols (``JitArgument`` and ``DynamicExpression``)
- Static shape wrapper functions for performance optimization
- Memory buffer management with configurable layouts
- Integration between PyTorch tensors and CuTe operations
- Modular kernel design through function composition

Key Integration Concepts
-----------------------

JitArgument Protocol Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports custom argument types through protocol implementation:

.. code-block:: python

   class BufferWithLayout:
       def __init__(self, ptr: cute.Pointer, stride_order: tuple[int, int, int]):
           self.ptr = ptr
           self.stride_order = stride_order
       
       # Implement JitArgument Protocol
       def __c_pointers__(self):
           """Get the C pointers for the underlying pointer."""
           return self.ptr.__c_pointers__()
       
       def __c_types__(self):
           """Get the C types for the pointer."""
           return self.ptr.__c_types__()
       
       # Implement DynamicExpression Protocol  
       def __mlir_type__(self):
           """Return the MLIR type for compilation."""
           return self.ptr.__mlir_type__()

**Protocol Benefits**:
- Seamless integration with CuTeDSL compilation pipeline
- Type-safe argument passing between host and device
- Automatic memory management and pointer handling

Dynamic Layout Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Flexible tensor layout creation based on runtime parameters:

.. code-block:: python

   class BufferWithLayout:
       def to_tensor(
           self, 
           shape: tuple[int, int, int], 
           *, 
           loc=None, 
           ip=None
       ) -> cute.Tensor:
           assert len(shape) == len(self.stride_order), (
               f"Shape {shape} and stride_order {self.stride_order} must have the "
               "same rank."
           )
           
           # Create layout with specified stride ordering
           layout = cute.make_ordered_layout(shape, self.stride_order)
           
           # Permute layout: (l, mn, k) -> (mn, k, l)
           res = cute.make_tensor(self.ptr, cute.select(layout, mode=[1, 2, 0]))
           return res

**Layout Features**:
- Runtime shape specification with compile-time stride ordering
- Layout permutation through ``cute.select()``
- Dimension reordering for optimal memory access patterns

JIT Function Composition
~~~~~~~~~~~~~~~~~~~~~~~

Modular kernel design through function composition:

.. code-block:: python

   @cute.jit
   def tensor_op_gemm_wrapper(
       buf_a: BufferWithLayout,
       buf_b: BufferWithLayout, 
       buf_c: BufferWithLayout,
       m: int,
       n: int,
       k: int,
       l: int,
   ):
       # Convert buffers to tensors with appropriate shapes
       tensor_a = buf_a.to_tensor((l, m, k))
       tensor_b = buf_b.to_tensor((l, n, k))
       tensor_c = buf_c.to_tensor((l, m, n))
       
       # Call specialized GEMM implementation
       gemm = TensorOpGemm(
           ab_dtype=cutlass.Float16,
           c_dtype=cutlass.Float16,
           acc_dtype=cutlass.Float32,
           cta_tiler=(128, 128, 32),
           atom_layout_mnk=(2, 2, 1),
           num_stages=3,
       )
       
       # Execute GEMM operation
       gemm(tensor_a, tensor_b, tensor_c)

**Composition Benefits**:
- Separation of memory management from computation logic
- Reusable buffer management across different kernels
- Clear interface boundaries between components

Static Shape Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL enables static shape specialization for performance:

.. code-block:: python

   # Compile with known shapes for optimization
   compiled_gemm = cute.compile(
       tensor_op_gemm_wrapper,
       buf_a_instance,
       buf_b_instance,
       buf_c_instance,
       M,  # Static values enable compile-time optimization
       N,
       K, 
       L,
   )

**Optimization Features**:
- Loop unrolling based on compile-time known bounds
- Elimination of dynamic branches and conditions
- Improved register allocation and instruction scheduling

Advanced Memory Management
-------------------------

Pointer Abstraction and Safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Safe pointer management through CuTeDSL abstractions:

.. code-block:: python

   def create_buffer_with_layout(tensor, stride_order):
       # Convert PyTorch tensor to CuTe pointer
       ptr = make_ptr(tensor.data_ptr(), tensor.dtype)
       
       # Wrap in custom buffer management class
       return BufferWithLayout(ptr, stride_order)
   
   # Usage with different stride orderings
   buf_a = create_buffer_with_layout(torch_tensor_a, (2, 0, 1))  # Batch-major
   buf_b = create_buffer_with_layout(torch_tensor_b, (1, 2, 0))  # Column-major

**Memory Safety**:
- Type-safe pointer creation from raw data pointers
- Automatic lifetime management through RAII patterns
- Consistent interface across different memory sources

Layout Transformation Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced layout manipulation for different access patterns:

.. code-block:: python

   def to_tensor(self, shape, *, loc=None, ip=None):
       # Create base layout with stride ordering
       layout = cute.make_ordered_layout(shape, self.stride_order)
       
       # Apply dimension permutation for optimal access
       # Original: (batch, M*N, K) 
       # Target:   (M*N, K, batch)
       permuted_layout = cute.select(layout, mode=[1, 2, 0])
       
       # Create tensor with transformed layout
       return cute.make_tensor(self.ptr, permuted_layout)

**Transformation Benefits**:
- Optimal memory access patterns for different algorithms
- Flexibility in data organization without memory copies
- Compile-time layout optimization

Integration Patterns
-------------------

PyTorch Integration
~~~~~~~~~~~~~~~~~~

Seamless integration with PyTorch tensors and operations:

.. code-block:: python

   def run_and_verify(M, N, K, L):
       # Create PyTorch tensors
       a = torch.randn(L, M, K, dtype=torch.float16, device="cuda")
       b = torch.randn(L, N, K, dtype=torch.float16, device="cuda") 
       c = torch.zeros(L, M, N, dtype=torch.float16, device="cuda")
       
       # Convert to CuTeDSL buffers
       buf_a = create_buffer_with_layout(a, (2, 0, 1))
       buf_b = create_buffer_with_layout(b, (1, 2, 0))
       buf_c = create_buffer_with_layout(c, (0, 1, 2))
       
       # Execute CuTeDSL operation
       compiled_gemm(buf_a, buf_b, buf_c, M, N, K, L)
       
       # Verify against PyTorch reference
       torch.testing.assert_close(c, torch.bmm(a, b.transpose(-1, -2)))

**Integration Features**:
- Zero-copy tensor conversion between frameworks
- Automatic device memory management
- Consistent numerical results with reference implementations

Modular Kernel Design
~~~~~~~~~~~~~~~~~~~~

Building complex operations from simpler components:

.. code-block:: python

   @cute.jit
   def complex_operation(buf_a, buf_b, buf_c, buf_d, shape_params):
       # Convert buffers to tensors
       tensor_a = buf_a.to_tensor(shape_params.a_shape)
       tensor_b = buf_b.to_tensor(shape_params.b_shape)
       
       # Intermediate computation
       intermediate = allocate_intermediate_buffer(shape_params.inter_shape)
       
       # Compose multiple operations
       gemm_op(tensor_a, tensor_b, intermediate)
       elementwise_op(intermediate, tensor_c)
       reduction_op(tensor_c, tensor_d)

**Design Principles**:
- Single responsibility per function
- Clear interfaces between components
- Composable building blocks for complex algorithms

Performance Considerations
-------------------------

Static vs Dynamic Shapes
~~~~~~~~~~~~~~~~~~~~~~~~

Strategic use of static and dynamic shapes for optimal performance:

.. code-block:: python

   # Static shapes for performance-critical inner loops
   @cute.jit
   def static_inner_kernel(tensor_a, tensor_b, tensor_c):
       # Shapes known at compile time - enables aggressive optimization
       pass
   
   # Dynamic shapes for flexible outer logic
   @cute.jit  
   def dynamic_wrapper(buf_a, buf_b, buf_c, runtime_shapes):
       # Runtime shape handling with minimal overhead
       tensor_a = buf_a.to_tensor(runtime_shapes.a)
       # Call static kernel with known shapes
       static_inner_kernel(tensor_a, tensor_b, tensor_c)

**Optimization Strategy**:
- Push dynamic logic to outer layers
- Maximize static optimization opportunities in inner kernels
- Use JIT compilation to eliminate runtime overhead

Memory Layout Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Strategic layout design for memory hierarchy efficiency:

.. code-block:: python

   # Optimal stride orderings for different access patterns
   BATCH_MAJOR = (2, 0, 1)    # Batch dimension has largest stride
   ROW_MAJOR = (0, 1, 2)      # Standard row-major ordering
   COL_MAJOR = (1, 0, 2)      # Column-major for transposed access
   
   # Choose layout based on algorithm requirements
   if algorithm_requires_transpose:
       buf_layout = COL_MAJOR
   elif batch_parallelism_important:
       buf_layout = BATCH_MAJOR
   else:
       buf_layout = ROW_MAJOR

Best Practices for JIT Integration
----------------------------------

1. **Protocol Implementation**: Implement both ``JitArgument`` and ``DynamicExpression`` protocols for custom types

2. **Memory Management**: Use RAII patterns and automatic lifetime management

3. **Layout Strategy**: Design layouts for optimal access patterns in target algorithms

4. **Shape Specialization**: Push static shape information as deep as possible for optimization

5. **Modular Design**: Separate memory management from computational logic

6. **Type Safety**: Leverage CuTeDSL's type system for compile-time error detection

7. **Integration Testing**: Validate results against reference implementations

8. **Performance Profiling**: Use static shapes for performance-critical code paths

This example demonstrates how CuTeDSL enables building sophisticated, reusable kernel components that integrate seamlessly with existing ML frameworks while maintaining high performance through careful design of memory management and compilation strategies.