# CuTe DSL Custom Passes and Communication Examples

This directory contains comprehensive examples demonstrating how to extend the CuTe DSL with custom passes and communication operations.

## Overview

The CuTe DSL (Domain Specific Language) provides a Python interface for writing GPU kernels that compile to efficient PTX code. This collection shows how to:

1. **Understand the lowering pipeline** from Python to PTX
2. **Add custom compilation passes** for optimization
3. **Implement communication primitives** for distributed computing
4. **Integrate custom MLIR passes** into the compilation pipeline

## Files Description

### Documentation

- **`cute_dsl_lowering_guide.md`**: Comprehensive guide explaining how CuTe DSL lowers to PTX code and how to add custom passes
- **`tutorial_custom_passes.md`**: Step-by-step tutorial for adding custom passes and communication commands

### Examples

- **`custom_evt_communication_pass.py`**: Example of creating a custom EVT (Epilogue Visitor Trees) pass for adding communication operations
- **`custom_cute_communication.py`**: Implementation of custom communication primitives using CuTe DSL and NVVM intrinsics  
- **`custom_mlir_pass_integration.py`**: Example of integrating custom MLIR passes into the compilation pipeline

## Quick Start

### 1. Understanding the Pipeline

The CuTe DSL compilation follows this flow:
```
Python DSL Code → CuTe MLIR Dialect → NVVM Dialect → PTX Code
```

Key pipeline: `cute-to-nvvm{cubin-format=bin}`

### 2. Adding a Simple EVT Pass

```python
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase

class MyCustomPass(EVTPassBase):
    dependencies = [PassShapeTypePropagation]
    
    def call(self):
        for node_meta in self.dag_ir.nodes_meta:
            # Your custom logic here
            pass
```

### 3. Creating Communication Primitives

```python
@dsl_user_op
def warp_allreduce_sum(value, *, loc=None, ip=None):
    warp_sync_op(loc=loc, ip=ip)
    result = value
    for offset in [16, 8, 4, 2, 1]:
        other = shuffle_sync_op(result, offset, loc=loc, ip=ip)
        result = result + other
    return result
```

### 4. Custom Pipeline Integration

```python
class CustomDSL(CutlassDSL):
    def _get_pipeline(self, pipeline):
        if pipeline is None:
            return (
                "builtin.module("
                "my-custom-pass,"
                "cute-to-nvvm{cubin-format=bin}"
                ")"
            )
        return super()._get_pipeline(pipeline)
```

## Running the Examples

### Prerequisites

1. CUTLASS with CuTe DSL support
2. CUDA Toolkit
3. Python 3.8+
4. Required packages (see `requirements.txt` in CuTeDSL directory)

### Example 1: EVT Communication Pass

```bash
cd examples/
python custom_evt_communication_pass.py
```

This example shows:
- Creating a custom EVT pass for communication
- Pass dependency management  
- DAG manipulation for adding communication nodes

### Example 2: Communication Primitives

```bash
python custom_cute_communication.py
```

This example demonstrates:
- Warp-level all-reduce, broadcast, all-gather
- Block-level communication using shared memory
- Cluster-level communication primitives
- Practical communication kernels

### Example 3: MLIR Pass Integration

```bash
python custom_mlir_pass_integration.py
```

This example covers:
- Custom DSL subclassing
- Pipeline string modification
- Pass manager integration
- Architecture-specific optimizations

## Key Concepts

### EVT Passes

EVT (Epilogue Visitor Trees) passes operate on computation graphs for epilogue operations:

- **Purpose**: Transform epilogue computation DAGs
- **Scope**: Epilogue fusion operations
- **Usage**: Matrix operations, element-wise operations

### MLIR Passes

MLIR passes operate at the intermediate representation level:

- **Purpose**: Lower-level optimizations and transformations
- **Scope**: Entire compilation pipeline
- **Usage**: Architecture-specific optimizations, custom dialects

### Communication Patterns

Supported communication patterns:

- **AllReduce**: Sum/Max/Min across all threads
- **Broadcast**: One-to-all communication
- **AllGather**: Gather from all threads
- **ReduceScatter**: Reduce and distribute

### Synchronization Scopes

- **Warp**: 32 threads (using shuffle operations)
- **Block**: All threads in a thread block (using shared memory)
- **Cluster**: Multiple thread blocks (using cluster shared memory)

## Best Practices

### Pass Development

1. **Dependencies**: Always specify correct pass dependencies
2. **Validation**: Add proper pre/post-condition checks
3. **Error Handling**: Provide meaningful error messages
4. **Performance**: Consider compilation time impact

### Communication Implementation

1. **Synchronization**: Always synchronize before communication
2. **Memory Access**: Minimize bank conflicts in shared memory
3. **Scalability**: Design for different scales (warp/block/cluster)
4. **Fallbacks**: Support older GPU architectures

### Testing

1. **Unit Tests**: Test individual passes and primitives
2. **Integration Tests**: Test complete pipelines
3. **Performance Tests**: Measure communication overhead
4. **Correctness Tests**: Validate communication semantics

## Advanced Topics

### Custom MLIR Dialect

For advanced users, you can create custom MLIR dialects:

```cpp
// my_dialect.h
class MyDialect : public mlir::Dialect {
    // Dialect definition
};

// my_pass.cpp  
class MyCustomPass : public mlir::PassWrapper<MyCustomPass, OperationPass<ModuleOp>> {
    void runOnOperation() override {
        // Pass implementation
    }
};
```

### Performance Optimization

Tips for optimizing custom passes:

1. **Pass Ordering**: Order passes for maximum benefit
2. **Analysis Caching**: Cache expensive analysis results
3. **Selective Application**: Only apply passes where beneficial
4. **Profiling**: Use profiling to identify bottlenecks

### Integration with Distributed Frameworks

Examples of integrating with distributed training frameworks:

```python
# PyTorch integration
class CutlassAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        # Use CuTe DSL all-reduce kernel
        return cutlass_allreduce_kernel(tensor)
    
    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output

# Usage in training loop
def training_step(model, data):
    output = model(data)
    loss = criterion(output, target)
    
    # Custom all-reduce in backward pass
    loss.backward()
    
    # Apply CuTe DSL all-reduce to gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad = CutlassAllReduce.apply(param.grad)
```

## Troubleshooting

### Common Issues

1. **"Pass X requires Y"**: Add missing dependencies
2. **"Unknown operation"**: Register custom operations with MLIR
3. **Race conditions**: Add proper synchronization
4. **Performance degradation**: Profile and optimize communication

### Debug Tips

1. Enable IR printing: `--enable-ir-printing`
2. Use debug assertions: `--enable-device-assertions`  
3. Profile kernels: Use `@cute.profile` decorator
4. Validate transformations: Print DAG before/after passes

## Contributing

When contributing improvements:

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Consider backward compatibility
5. Profile performance impact

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CuTe Programming Guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_cute_overview.md)
- [MLIR Documentation](https://mlir.llvm.org/)
- [NVVM IR Specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [PTX ISA Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## License

These examples are provided under the same license as CUTLASS. See the main repository for license details.