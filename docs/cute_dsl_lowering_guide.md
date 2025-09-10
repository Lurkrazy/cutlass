# CuTe DSL Lowering to PTX Guide

This guide explains how the CuTe DSL (Domain Specific Language) lowers to PTX code and how to add custom passes or communication commands to the compilation pipeline.

## Overview of the Lowering Pipeline

The CuTe DSL uses MLIR (Multi-Level Intermediate Representation) to compile Python DSL code to PTX. The compilation flow follows this path:

```
Python DSL Code → CuTe MLIR Dialect → NVVM Dialect → PTX Code
```

### Key Components

1. **CuTe DSL Frontend** (`python/CuTeDSL/cutlass_dsl/cutlass.py`)
   - Translates Python code to CuTe MLIR dialect
   - Manages the compilation pipeline
   - Handles GPU module generation

2. **MLIR Pass Manager** (`python/CuTeDSL/base_dsl/compiler.py`)
   - Executes the lowering passes
   - Manages compilation options
   - Handles error reporting

3. **NVVM Wrappers** (`python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py`)
   - Provides Python bindings for NVVM intrinsics
   - Enables direct PTX instruction generation

## The Compilation Pipeline

### Default Pipeline

The main compilation pipeline is defined in `cutlass_dsl/cutlass.py`:

```python
def _get_pipeline(self, pipeline):
    if pipeline == None:
        return (
            "builtin.module(cute-to-nvvm{cubin-format=bin "
            + self.compile_options.to_str()
            + "})"
        )
    return pipeline
```

### Pipeline Stages

1. **cute-to-nvvm**: Main lowering pass that converts CuTe dialect to NVVM
2. **external-kernel-for-gpu-launch**: Adds support for GPU kernel execution
3. **Additional passes**: Can be added based on compilation options

## How to Add Custom Passes

### Method 1: EVT (Epilogue Visitor Trees) Passes

For epilogue fusion operations, you can add custom passes by extending the EVT pass system:

#### Step 1: Create a Custom EVT Pass

Create a new file in `python/cutlass/backend/evt/passes/`:

```python
# python/cutlass/backend/evt/passes/pass_custom_communication.py

from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation

class PassCustomCommunication(EVTPassBase):
    """
    Custom pass for adding communication operations to the epilogue.
    This pass can add collective operations like AllReduce, AllGather, etc.
    """
    dependencies = [
        PassShapeTypePropagation,  # Ensure shape/type info is available
    ]

    def requires(self) -> None:
        """Verify required nodes exist in the DAG"""
        # Add any precondition checks here
        pass

    def call(self):
        """Main pass logic"""
        # Example: Add communication nodes to the DAG
        for node_meta in self.dag_ir.nodes_meta:
            if node_meta.op == "compute":
                # Add custom communication logic here
                self._add_communication_node(node_meta)

    def _add_communication_node(self, node_meta):
        """Add a communication node after a compute operation"""
        # Implementation depends on the specific communication pattern
        # This could add AllReduce, broadcast, or other collective operations
        pass

    def ensures(self) -> None:
        """Post-pass validation"""
        # Verify the pass completed successfully
        pass
```

#### Step 2: Register the Pass

Add your pass to the pass list in `python/cutlass/backend/evt/passes/__init__.py`:

```python
from cutlass_cppgen.backend.evt.passes.pass_custom_communication import PassCustomCommunication
```

#### Step 3: Use the Pass

Include your pass in the compilation pipeline:

```python
from cutlass_cppgen.backend.evt.passes import PassCustomCommunication

# Add to your pass list
pass_list = [
    PassShapeTypePropagation,
    PassFixElementD,
    PassCustomCommunication,  # Your custom pass
    PassGetImpl,
]

pass_manager = EVTPassManager(dag_ir, pass_list)
pass_manager()
```

### Method 2: MLIR Passes

For lower-level transformations in the CuTe DSL pipeline:

#### Step 1: Modify the Pipeline String

You can add custom MLIR passes by modifying the pipeline string:

```python
class CustomCutlassDSL(CutlassDSL):
    def _get_pipeline(self, pipeline):
        if pipeline is None:
            # Add your custom passes to the pipeline
            return (
                "builtin.module("
                "custom-communication-pass,"  # Your custom pass
                "cute-to-nvvm{cubin-format=bin "
                + self.compile_options.to_str()
                + "})"
            )
        return super()._get_pipeline(pipeline)
```

#### Step 2: Create Custom Pass (C++/MLIR)

If you need to implement a new MLIR pass, you would typically:

1. Implement the pass in C++ using MLIR's pass infrastructure
2. Register it with the MLIR pass registry
3. Make it available in the pipeline

Note: This requires modifying the C++ codebase and rebuilding the MLIR library.

## Adding Custom Communication Commands

### Method 1: High-Level Communication Primitives

Add communication operations using NVVM intrinsics:

```python
# python/CuTeDSL/cutlass/cute/arch/communication.py

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch.nvvm_wrappers import warp_sync_op

@dsl_user_op
def allreduce_warp(value, *, loc=None, ip=None):
    """
    Perform a warp-level all-reduce operation.
    """
    # Synchronize the warp first
    warp_sync_op(loc=loc, ip=ip)
    
    # Implement butterfly reduction pattern
    result = value
    for offset in [16, 8, 4, 2, 1]:
        other = shuffle_sync_op(result, offset, loc=loc, ip=ip)
        result = result + other
    
    return result

@dsl_user_op
def broadcast_warp(value, src_lane=0, *, loc=None, ip=None):
    """
    Broadcast a value from src_lane to all threads in the warp.
    """
    warp_sync_op(loc=loc, ip=ip)
    return shuffle_sync_op(value, src_lane, loc=loc, ip=ip)

@dsl_user_op
def barrier_cluster(*, loc=None, ip=None):
    """
    Synchronize all threads in a cluster.
    """
    nvvm.barrier_cluster_arrive(loc=loc, ip=ip)
    nvvm.barrier_cluster_wait(loc=loc, ip=ip)
```

### Method 2: Using Existing NVVM Wrappers

Leverage the existing NVVM wrappers for communication:

```python
from cutlass.cute.arch.nvvm_wrappers import (
    warp_sync_op,
    shuffle_sync_op,
    fence_acq_rel_cta,
    fence_acq_rel_cluster,
    cp_async_commit_group,
    cp_async_wait_group
)

@cute.jit
def communication_example(data: TensorView):
    # Warp-level communication
    tid = thread_idx()
    if tid < 32:  # First warp
        value = data[tid]
        
        # All-reduce within warp
        warp_sync_op()
        for offset in [16, 8, 4, 2, 1]:
            other = shuffle_sync_op(value, offset)
            value = value + other
        
        # Broadcast result to all threads
        result = shuffle_sync_op(value, 0)
        data[tid] = result
    
    # Block-level synchronization
    fence_acq_rel_cta()
    
    # Cluster-level synchronization (if available)
    fence_acq_rel_cluster()
```

## Practical Examples

### Example 1: Custom Epilogue Pass for Gradient Communication

```python
# Custom pass for adding gradient all-reduce in epilogue
class PassGradientAllReduce(EVTPassBase):
    dependencies = [PassShapeTypePropagation, PassFixElementD]
    
    def call(self):
        # Find gradient computation nodes
        for node_meta in self.dag_ir.nodes_meta:
            if self._is_gradient_node(node_meta):
                # Insert all-reduce after gradient computation
                self._insert_allreduce(node_meta)
    
    def _is_gradient_node(self, node_meta):
        # Logic to identify gradient computation nodes
        return node_meta.op == "compute" and "grad" in node_meta.name
    
    def _insert_allreduce(self, node_meta):
        # Add all-reduce operation
        # This would modify the DAG to include communication
        pass
```

### Example 2: Custom Communication Kernel

```python
@cute.jit
def custom_communication_kernel(
    input_data: TensorView,
    output_data: TensorView,
    comm_pattern: CommPattern
):
    """
    Custom kernel with embedded communication operations.
    """
    tid = thread_idx()
    bid = block_idx()
    
    # Load data
    local_data = input_data[tid]
    
    # Apply communication pattern
    if comm_pattern == CommPattern.ALLREDUCE:
        # Warp-level all-reduce
        warp_sync_op()
        for offset in [16, 8, 4, 2, 1]:
            other = shuffle_sync_op(local_data, offset)
            local_data = local_data + other
        
        # Store back the reduced value
        output_data[tid] = shuffle_sync_op(local_data, 0)
    
    elif comm_pattern == CommPattern.ALLGATHER:
        # Implementation for all-gather
        warp_sync_op()
        for i in range(32):
            gathered_value = shuffle_sync_op(local_data, i)
            output_data[tid * 32 + i] = gathered_value
```

## Best Practices

1. **Understand Dependencies**: Always specify pass dependencies correctly in the EVT system
2. **Validation**: Add proper validation in `requires()` and `ensures()` methods
3. **Error Handling**: Handle edge cases and provide meaningful error messages
4. **Performance**: Consider the impact of your passes on compilation time and runtime performance
5. **Testing**: Create comprehensive tests for your custom passes and communication operations

## Debugging and Development

### Viewing MLIR IR

To debug the lowering pipeline, you can examine the MLIR IR at different stages:

```python
# Enable IR printing during compilation
dsl = CutlassDSL()
dsl.compile_options = CompileOptions("--enable-device-assertions")

# The IR will be printed during compilation
```

### Custom Pipeline Development

For development, you can create a custom pipeline with additional debugging passes:

```python
def debug_pipeline():
    return (
        "builtin.module("
        "print-ir{print-after-all},"  # Print IR after each pass
        "cute-to-nvvm{cubin-format=bin},"
        "print-ir{print-after-all}"
        ")"
    )
```

This guide provides a comprehensive overview of how CuTe DSL lowers to PTX and how to extend the system with custom passes and communication operations.