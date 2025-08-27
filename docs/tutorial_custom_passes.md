# Tutorial: Adding Custom Passes and Communication Commands to CuTe DSL

This tutorial provides step-by-step instructions for extending the CuTe DSL with custom passes and communication operations.

## Table of Contents

1. [Understanding the Compilation Pipeline](#understanding-the-compilation-pipeline)
2. [Adding EVT Passes](#adding-evt-passes)  
3. [Adding MLIR Passes](#adding-mlir-passes)
4. [Creating Communication Primitives](#creating-communication-primitives)
5. [Practical Examples](#practical-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Understanding the Compilation Pipeline

### Overview

The CuTe DSL compilation pipeline transforms Python code into PTX through several stages:

```
Python DSL → AST → CuTe MLIR → NVVM MLIR → PTX
```

Key components:
- **Frontend**: Converts Python AST to CuTe MLIR dialect
- **Pass Manager**: Applies transformation passes  
- **Backend**: Lowers to NVVM and generates PTX

### Pipeline Configuration

The default pipeline is defined in `cutlass_dsl/cutlass.py`:

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

## Adding EVT Passes

EVT (Epilogue Visitor Trees) passes operate on the computation graph for epilogue operations.

### Step 1: Create Your Pass

Create a new file `python/cutlass/backend/evt/passes/pass_my_custom.py`:

```python
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation

class PassMyCustom(EVTPassBase):
    """
    Description of what your pass does.
    """
    dependencies = [PassShapeTypePropagation]
    
    def requires(self) -> None:
        # Validate preconditions
        pass
    
    def call(self):
        # Main pass logic
        for node_meta in self.dag_ir.nodes_meta:
            # Process each node
            self.process_node(node_meta)
    
    def process_node(self, node_meta):
        # Your custom logic here
        pass
    
    def ensures(self) -> None:
        # Validate postconditions
        pass
```

### Step 2: Register the Pass

Add to `python/cutlass/backend/evt/passes/__init__.py`:

```python
from cutlass_cppgen.backend.evt.passes.pass_my_custom import PassMyCustom
```

### Step 3: Use the Pass

```python
from cutlass_cppgen.backend.evt.passes import (
    EVTPassManager,
    PassShapeTypePropagation,
    PassMyCustom
)

pass_list = [
    PassShapeTypePropagation,
    PassMyCustom,
    # ... other passes
]

pass_manager = EVTPassManager(dag_ir, pass_list)
pass_manager()
```

## Adding MLIR Passes

For lower-level transformations in the compilation pipeline:

### Method 1: Pipeline String Modification

```python
class CustomDSL(CutlassDSL):
    def _get_pipeline(self, pipeline):
        if pipeline is None:
            return (
                "builtin.module("
                "my-custom-pass{option1=value1},"
                "cute-to-nvvm{cubin-format=bin "
                + self.compile_options.to_str()
                + "})"
            )
        return super()._get_pipeline(pipeline)
```

### Method 2: Pass Manager Integration

```python
class PassManager:
    def __init__(self):
        self.passes = []
    
    def add_pass(self, pass_name, options=None):
        pass_config = {"name": pass_name}
        if options:
            pass_config.update(options)
        self.passes.append(pass_config)
    
    def build_pipeline(self, base_pipeline):
        # Insert custom passes into pipeline
        pass
```

## Creating Communication Primitives

### Warp-Level Communication

```python
@dsl_user_op
def warp_allreduce(value, *, loc=None, ip=None):
    """All-reduce within a warp."""
    warp_sync_op(loc=loc, ip=ip)
    
    result = value
    for offset in [16, 8, 4, 2, 1]:
        other = shuffle_sync_op(result, offset, loc=loc, ip=ip)
        result = result + other
    
    return result
```

### Block-Level Communication

```python
@dsl_user_op  
def block_allreduce(value, shared_mem, *, loc=None, ip=None):
    """All-reduce within a thread block."""
    # Reduce within warps first
    warp_result = warp_allreduce(value, loc=loc, ip=ip)
    
    # Store warp results in shared memory
    warp_id_val = warp_id(loc=loc, ip=ip)
    lane_id_val = lane_id(loc=loc, ip=ip)
    
    if lane_id_val == 0:
        shared_mem[warp_id_val] = warp_result
    
    # Block synchronization
    fence_acq_rel_cta(loc=loc, ip=ip)
    
    # Reduce across warps
    if warp_id_val == 0:
        warp_data = shared_mem[lane_id_val] if lane_id_val < 32 else 0
        final_result = warp_allreduce(warp_data, loc=loc, ip=ip)
        if lane_id_val == 0:
            shared_mem[0] = final_result
    
    fence_acq_rel_cta(loc=loc, ip=ip)
    return shared_mem[0]
```

### Cluster-Level Communication

```python
@dsl_user_op
def cluster_broadcast(value, src_block=0, *, loc=None, ip=None):
    """Broadcast across thread block cluster."""
    block_id = cute.block_idx_x()
    
    # Cluster synchronization
    fence_acq_rel_cluster(loc=loc, ip=ip)
    
    # Implementation depends on cluster shared memory
    # or other coordination mechanisms
    return value
```

## Practical Examples

### Example 1: Gradient All-Reduce Pass

```python
class PassGradientAllReduce(EVTPassBase):
    def call(self):
        gradient_nodes = self.find_gradient_nodes()
        
        for node in gradient_nodes:
            # Insert all-reduce after gradient computation
            self.insert_allreduce_after(node)
    
    def find_gradient_nodes(self):
        return [n for n in self.dag_ir.nodes_meta 
                if 'grad' in n.name.lower()]
    
    def insert_allreduce_after(self, node):
        # Create communication node
        comm_node = self.create_communication_node(
            f"{node.name}_allreduce",
            "allreduce",
            node.output_tensor
        )
        # Add to DAG
        self.dag_ir.add_node(comm_node)
        self.dag_ir.add_edge(node.name, comm_node.name)
```

### Example 2: Custom Communication Kernel

```python
@cute.jit
def distributed_training_kernel(
    gradients: cute.TensorView,
    parameters: cute.TensorView,
    learning_rate: float,
    world_size: int
):
    """Kernel with built-in gradient all-reduce."""
    tid = thread_idx()
    
    # Load gradient
    grad = gradients[tid]
    
    # All-reduce gradients across processes
    reduced_grad = warp_allreduce(grad)
    
    # Average by world size
    avg_grad = reduced_grad / world_size
    
    # Update parameters
    param = parameters[tid]
    updated_param = param - learning_rate * avg_grad
    parameters[tid] = updated_param
```

### Example 3: Custom Pipeline for Communication

```python
class CommunicationOptimizedDSL(CutlassDSL):
    def __init__(self, comm_type="allreduce", **kwargs):
        super().__init__(**kwargs)
        self.comm_type = comm_type
    
    def _get_pipeline(self, pipeline):
        if pipeline is None:
            return (
                "builtin.module("
                f"communication-analysis{{type={self.comm_type}}},"
                "communication-optimization,"
                "cute-to-nvvm{cubin-format=bin "
                + self.compile_options.to_str()
                + "})"
            )
        return super()._get_pipeline(pipeline)

# Usage
dsl = CommunicationOptimizedDSL(comm_type="allreduce")

@dsl.jit
def optimized_kernel(data):
    # Kernel will be optimized for all-reduce operations
    result = cute.allreduce_sum(data)
    return result
```

## Best Practices

### 1. Pass Design

- **Single Responsibility**: Each pass should have one clear purpose
- **Dependencies**: Clearly specify pass dependencies
- **Validation**: Add proper pre/post-condition checks
- **Error Handling**: Provide meaningful error messages

### 2. Communication Patterns

- **Synchronization**: Always synchronize before communication
- **Memory Access**: Minimize shared memory bank conflicts
- **Scalability**: Consider performance at different scales (warp/block/cluster)
- **Fallbacks**: Provide fallback implementations for older architectures

### 3. Performance Considerations

- **Pass Ordering**: Order passes for optimal performance
- **Compilation Time**: Minimize expensive analysis passes
- **Runtime Overhead**: Avoid unnecessary synchronization
- **Memory Usage**: Optimize shared memory utilization

### 4. Testing

```python
def test_custom_pass():
    # Create test DAG
    dag_ir = create_test_dag()
    
    # Apply pass
    pass_instance = PassMyCustom(dag_ir)
    pass_instance()
    
    # Validate results
    assert validate_dag_transformation(dag_ir)

def test_communication_primitive():
    # Test with known inputs
    input_data = np.array([1, 2, 3, 4], dtype=np.float32)
    expected_sum = np.sum(input_data)
    
    # Run kernel
    result = run_allreduce_kernel(input_data)
    
    # Validate
    assert np.allclose(result, expected_sum)
```

## Troubleshooting

### Common Issues

1. **Pass Dependency Errors**
   ```
   Error: Pass X requires Y but Y was not run
   ```
   **Solution**: Add Y to the dependencies list of X

2. **MLIR Compilation Errors**
   ```
   Error: unknown operation 'custom.my_op'
   ```
   **Solution**: Ensure your custom operations are registered with MLIR

3. **Runtime Synchronization Issues**
   ```
   Error: Race condition in shared memory access
   ```
   **Solution**: Add proper synchronization barriers

4. **Performance Degradation**
   ```
   Warning: Communication overhead exceeds computation
   ```
   **Solution**: Optimize communication patterns or increase computation intensity

### Debugging Tips

1. **Enable IR Printing**:
   ```python
   compile_options = CompileOptions("--enable-ir-printing")
   ```

2. **Use Debug Assertions**:
   ```python
   compile_options = CompileOptions("--enable-device-assertions")
   ```

3. **Profile Communication**:
   ```python
   @cute.profile
   def communication_kernel():
       # Your kernel code
       pass
   ```

4. **Validate Pass Output**:
   ```python
   def debug_pass(dag_ir):
       print(f"Nodes: {len(dag_ir.nodes_meta)}")
       for node in dag_ir.nodes_meta:
           print(f"  {node.name}: {node.op}")
   ```

## Next Steps

1. Study the existing EVT passes for patterns and best practices
2. Experiment with simple communication primitives first
3. Gradually build more complex passes and optimizations
4. Contribute your improvements back to the community

For more advanced topics, see:
- [MLIR Pass Infrastructure Documentation](https://mlir.llvm.org/docs/PassInfrastructure/)
- [NVVM IR Documentation](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [CuTe Programming Guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_cute_overview.md)