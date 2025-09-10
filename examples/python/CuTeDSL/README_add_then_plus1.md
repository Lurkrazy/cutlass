# AddThenPlus1 Custom Operator Integration

This directory contains the implementation of a custom operator called `AddThenPlus1` that integrates with the CuTe DSL lowering pipeline.

## Operation

The `AddThenPlus1` operator computes:
```
out[i] = a[i] + b[i] + 1
```

## Files

### C++ Implementation
- **`include/cutlass/epilogue/thread/add_then_plus1.h`** - CUTLASS epilogue functor implementing the AddThenPlus1 operation

### Python DSL Integration  
- **`examples/python/CuTeDSL/ops/add_then_plus1.py`** - Python DSL operator implementation
- **`examples/python/CuTeDSL/ampere/vector_add_then_plus1.py`** - Example script demonstrating usage

### Tests
- **`tests/test_add_then_plus1.py`** - Comprehensive pytest test suite

## Usage

### Running the Example

```bash
cd /path/to/cutlass

# Basic example with default parameters (1024x1024)
python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py

# Specify custom dimensions
python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 512 --N 256

# Run with benchmarking
python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 2048 --N 2048 --benchmark --iterations 1000
```

### Running Tests

```bash
# Run all AddThenPlus1 tests
pytest tests/test_add_then_plus1.py -v

# Run specific test
pytest tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_small -v
```

## Verification

The example script performs the following verification:

1. **Baseline computation**: `ref[i] = a[i] + b[i]`
2. **Custom operator computation**: `res[i] = AddThenPlus1(a[i], b[i])`  
3. **Assertion check**: `res[i] == ref[i] + 1` (elementwise)

The test outputs "PASS" when the verification succeeds.

## Integration Points

The implementation demonstrates two key integration points in the CuTe DSL pipeline:

1. **Epilogue Functor Level**: The C++ `AddThenPlus1` epilogue functor can be used with GEMM operations and other CUTLASS kernels
2. **Python DSL Level**: The Python operator integrates with the CuTe DSL compilation and execution pipeline

## Architecture Support

- **Target Architecture**: Ampere GPUs (compute capability 8.0+)
- **Data Type**: Float32 (can be extended to other types)
- **Memory Layout**: Row-major tensors with dynamic layout support

## Performance

The implementation includes:
- Vectorized memory accesses (128-bit by default)
- Predicated memory operations for out-of-bounds safety
- Optimized thread-block tiling patterns
- Benchmarking utilities with memory throughput calculations

## Extension

This implementation serves as a template for creating additional custom operators. Key patterns include:

- Custom epilogue functor structure in C++
- CuTe DSL kernel implementation patterns
- Tensor partitioning and tiling strategies
- Memory access optimization techniques