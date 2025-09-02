.. _example_elementwise_apply:

Elementwise Apply Example  
=========================

The elementwise apply example demonstrates CuTeDSL's meta-programming capabilities by allowing runtime customization of elementwise operations through lambda functions. This showcases how to parameterize CUDA kernels with operation types at compile time.

Overview
--------

**File**: ``examples/python/CuTeDSL/ampere/elementwise_apply.py``

**Purpose**: Apply arbitrary binary operations to tensor elements using configurable lambda functions while demonstrating:

- Meta-programming with compile-time function parameters
- Dynamic operation selection using ``cutlass.Constexpr``
- List comprehensions and iterations over compile-time known collections
- Stream-based asynchronous kernel execution
- Advanced TV layout optimization strategies

Key CuTeDSL Meta-Programming Concepts
------------------------------------

Compile-Time Constants and Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports compile-time parameterization through ``cutlass.Constexpr``:

.. code-block:: python

   @cute.kernel
   def elementwise_apply_kernel(
       op: cutlass.Constexpr,           # Compile-time operation parameter
       inputs: List[cute.Tensor],       # Dynamic list of input tensors
       gC: cute.Tensor,
       cC: cute.Tensor,
       shape: cute.Shape,
       tv_layout: cute.Layout,
   ):
       # The 'op' parameter is resolved at compile time
       result = op(*[frgInput.load() for frgInput in frgInputs])

**Key Points**:
- ``cutlass.Constexpr`` parameters are resolved during JIT compilation
- Operations passed as lambda functions get inlined into the generated kernel
- This enables type-safe operation customization without runtime overhead

Dynamic Collection Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL provides utilities for iterating over compile-time known collections:

.. code-block:: python

   # Iterate over input tensors with compile-time unrolling
   ctaInputs = [t[cta_coord] for t in inputs]  # List comprehension
   
   # Print loop that unrolls at compile time
   for i in cutlass.range_constexpr(len(ctaInputs)):
       print(f"[DSL INFO]   ctaInputs{i} = {ctaInputs[i].type}")
   
   # Zip iteration over tensors and fragments
   for thrInput, frgInput in zip(thrInputs, frgInputs):
       cute.copy(copy_atom_load, thrInput, frgInput, pred=frgPred)

**Collection Operations**:
- ``cutlass.range_constexpr()``: Compile-time range for loop unrolling
- List comprehensions automatically unroll when applied to compile-time known collections
- ``zip()`` operations work seamlessly with DSL collections

Advanced Layout Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example demonstrates sophisticated TV layout strategies:

.. code-block:: python

   # Strategy 1: Optimized 2D thread layout for coalesced access
   thr_layout = cute.make_layout((4, 32), stride=(32, 1))
   val_layout = cute.make_layout((4, 4), stride=(4, 1))
   tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

The comments in the code show various layout strategies:

.. code-block:: python

   # Baseline: naive TV layout - poor performance
   # tv_layout = cute.make_layout((128, (4, 4)), stride=(4, (512, 1)))
   
   # Opt-1: better 1D thread layout for coalesced access
   # tv_layout = cute.make_layout((128, (4, 4)), stride=(16, (4, 1)))
   
   # Opt-2: 2D tile with suboptimal vectorization
   # tv_layout = cute.make_layout(((32, 4), (4, 4)), stride=((4, 512), (1, 128)))
   
   # Opt-3: SOL (Solution) with 2D thread tile - optimal performance
   thr_layout = cute.make_layout((4, 32), stride=(32, 1))
   val_layout = cute.make_layout((4, 4), stride=(4, 1))

**Layout Design Principles**:
- Thread IDs should map to contiguous memory addresses for coalescing
- Value layouts should enable vectorized memory operations
- Right-most dimensions should have stride-1 for optimal vector operations

Stream and Asynchronous Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports CUDA streams for asynchronous execution:

.. code-block:: python

   @cute.jit
   def elementwise_apply(
       op: cutlass.Constexpr,
       a: cute.Tensor,
       b: cute.Tensor, 
       result: cute.Tensor,
       stream: cuda.CUstream,  # CUDA stream parameter
   ):
       # Launch kernel on specified stream
       elementwise_apply_kernel(...).launch(
           grid=[cute.size(gC, mode=[1]), 1, 1],
           block=[cute.size(tv_layout, mode=[0]), 1, 1],
           stream=stream,  # Stream-based execution
       )

**Stream Integration**:
- Kernels can be launched on custom CUDA streams
- Enables overlap with host computation and other GPU work
- Supports integration with PyTorch's stream management

Compile-Time Validation and Assertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuTeDSL supports compile-time validation through assertions:

.. code-block:: python

   # Compile-time validation of tensor element types
   assert all(t.element_type == inputs[0].element_type for t in inputs)
   
   # This assertion is checked during JIT compilation
   # If failed, compilation will fail with a clear error message

Advanced Memory Copy Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example shows sophisticated copy atom configuration:

.. code-block:: python

   copy_atom_load = cute.make_copy_atom(
       cute.nvgpu.CopyUniversalOp(),
       inputs[0].element_type,
       num_bits_per_copy=inputs[0].element_type.width,  # Single element copy
   )
   copy_atom_store = cute.make_copy_atom(
       cute.nvgpu.CopyUniversalOp(),
       gC.element_type,
       num_bits_per_copy=gC.element_type.width,  # Single element copy
   )

**Copy Configuration**:
- ``num_bits_per_copy`` controls vectorization level
- ``CopyUniversalOp`` provides flexible memory operation support
- Copy atoms can be configured per element type and access pattern

Meta-Programming Usage Patterns
-------------------------------

Lambda Function Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example usage with different operations
   elementwise_apply(operator.add, tensor_a, tensor_b, result)
   elementwise_apply(operator.mul, tensor_a, tensor_b, result) 
   elementwise_apply(lambda a, b: a * a + b * b, tensor_a, tensor_b, result)

**Operation Examples**:
- Built-in operators: ``operator.add``, ``operator.mul``, ``operator.sub``
- Custom lambda functions: ``lambda a, b: a * a + b * b``
- Complex expressions: ``lambda a, b: cute.max(a, b) + cute.abs(a - b)``

Compilation and Caching
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compile with inlined operation
   compiled_func = cute.compile(
       elementwise_apply,
       op,                    # Operation gets inlined
       from_dlpack(a),
       from_dlpack(b), 
       from_dlpack(c).mark_layout_dynamic(),
       current_stream,
   )
   
   # Subsequent calls use cached compiled version
   compiled_func(tensor_a, tensor_b, result_tensor, stream)

**Compilation Benefits**:
- Operation is fully inlined - no function call overhead
- Type specialization occurs during compilation
- Cached compiled functions can be reused efficiently

Benchmarking Integration
~~~~~~~~~~~~~~~~~~~~~~~

The example demonstrates integration with CuTeDSL's benchmarking framework:

.. code-block:: python

   avg_time_us = testing.benchmark(
       compiled_func,
       kernel_arguments=testing.JitArguments(
           from_dlpack(a),
           from_dlpack(b),
           from_dlpack(c).mark_layout_dynamic(),
           current_stream,
       ),
       warmup_iterations=warmup_iterations,
       iterations=iterations,
       use_cuda_graphs=True,     # Enable CUDA graphs for performance
       stream=current_stream,
   )

**Benchmarking Features**:
- ``testing.JitArguments`` for argument packaging
- CUDA graphs support for reduced launch overhead
- Stream-aware benchmarking
- Automatic warmup and measurement iteration handling

Best Practices for Meta-Programming
-----------------------------------

1. **Constexpr Usage**: Mark compile-time parameters with ``cutlass.Constexpr`` for optimal performance

2. **Collection Unrolling**: Use ``cutlass.range_constexpr()`` for compile-time known loop bounds

3. **Type Validation**: Add compile-time assertions to validate tensor compatibility

4. **Stream Management**: Leverage CUDA streams for optimal GPU utilization

5. **Operation Inlining**: Pass operations as lambda functions for zero-overhead customization

6. **Layout Optimization**: Design TV layouts with memory coalescing in mind

7. **Compilation Caching**: Compile operations once and reuse for repeated calls

This example demonstrates how CuTeDSL's meta-programming capabilities enable flexible, high-performance kernel customization while maintaining the benefits of static compilation and optimization.