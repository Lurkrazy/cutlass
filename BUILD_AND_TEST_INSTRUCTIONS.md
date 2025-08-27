# Build and Test Instructions for AddThenPlus1 Custom Operator

This document provides step-by-step instructions for building and testing the AddThenPlus1 custom operator integration with the CuTe DSL.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 8.0+ (Ampere architecture or later)
- CUDA Toolkit 11.0+

### Software Requirements
- CMake 3.18+
- GCC 7.5+ or Clang 6.0+
- Python 3.7+
- PyTorch with CUDA support
- pytest (for running tests)

## Building CUTLASS with CuTe DSL Support

1. **Clone and prepare the repository:**
   ```bash
   git clone https://github.com/NVIDIA/cutlass.git
   cd cutlass
   git submodule update --init --recursive
   ```

2. **Build CUTLASS with Python support:**
   ```bash
   mkdir build && cd build
   cmake .. -DCUTLASS_ENABLE_PYTHON=ON -DCUTLASS_ENABLE_CUTEDSL=ON
   make -j$(nproc)
   ```

3. **Install Python packages:**
   ```bash
   cd ../python
   pip install -e .
   ```

## Testing the AddThenPlus1 Implementation

### Environment Setup

1. **Verify CUDA environment:**
   ```bash
   python -c "
   import torch
   print(f'PyTorch version: {torch.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   if torch.cuda.is_available():
       print(f'CUDA device: {torch.cuda.get_device_name(0)}')
       print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
   "
   ```

2. **Verify CuTe DSL installation:**
   ```bash
   python -c "
   import cutlass
   import cutlass.cute as cute
   print('✓ CuTe DSL imported successfully')
   "
   ```

### Running the Example

1. **Basic functionality test:**
   ```bash
   cd /path/to/cutlass
   python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 1024 --N 512
   ```

2. **Small test for debugging:**
   ```bash
   python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 32 --N 64
   ```

3. **Performance benchmark:**
   ```bash
   python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 2048 --N 2048 --benchmark --iterations 1000
   ```

### Running Unit Tests

1. **Run all AddThenPlus1 tests:**
   ```bash
   pytest tests/test_add_then_plus1.py -v
   ```

2. **Run specific test cases:**
   ```bash
   # Test with small tensors
   pytest tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_small -v
   
   # Test with 2D tensors
   pytest tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_2d -v
   
   # Test with specific values
   pytest tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_specific_values -v
   ```

## Expected Output

### Successful Example Run
```
=== AddThenPlus1 Custom Operator Test ===
Tensor dimensions: [1024, 512]
Data type: cutlass.Float32
Operation: out[i] = a[i] + b[i] + 1

Input tensor shapes:
a: torch.Size([1024, 512]), dtype: torch.float32
b: torch.Size([1024, 512]), dtype: torch.float32
c: torch.Size([1024, 512]), dtype: torch.float32

Compiling AddThenPlus1 kernel with cute.compile ...
Compilation time: 2.3456 seconds

Executing custom AddThenPlus1 kernel...
Verifying results...
Max absolute error: 0.0
✓ PASS: AddThenPlus1 works correctly!
✓ Verified: res[i] == (a[i] + b[i]) + 1

🎉 All tests passed! AddThenPlus1 integration successful.
```

### Successful Test Run
```
pytest tests/test_add_then_plus1.py -v

tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_small PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_2d PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_specific_values PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_zeros PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_negative_values PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_various_sizes[64-128] PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_various_sizes[128-256] PASSED
tests/test_add_then_plus1.py::TestAddThenPlus1::test_add_then_plus1_various_sizes[256-512] PASSED

========================== 8 passed in 15.42s ==========================
```

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   - Ensure NVIDIA drivers and CUDA toolkit are properly installed
   - Verify GPU compute capability meets requirements (8.0+)

2. **Import errors:**
   - Ensure CUTLASS Python package is installed: `pip install -e python/`
   - Verify PYTHONPATH includes the CUTLASS Python directory

3. **Compilation errors:**
   - Ensure CMake was configured with `-DCUTLASS_ENABLE_PYTHON=ON`
   - Check that all submodules are updated: `git submodule update --init --recursive`

4. **Memory errors:**
   - Reduce tensor sizes if running out of GPU memory
   - Use smaller batch sizes for testing

### Performance Profiling

To profile the AddThenPlus1 kernel with NVIDIA Nsight Compute:

```bash
ncu --set full python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 2048 --N 2048 --iterations 10 --skip_ref_check
```

## Integration Verification

The implementation successfully demonstrates:

1. **C++ Epilogue Functor**: `cutlass/epilogue/thread/add_then_plus1.h` implements the CUTLASS epilogue interface
2. **Python DSL Integration**: `ops/add_then_plus1.py` provides CuTe DSL kernel implementation
3. **Lowering Pipeline**: The operator integrates with the CuTe DSL → NVVM → PTX compilation flow
4. **Correctness**: Tests verify `res[i] == (a[i] + b[i]) + 1` for all elements
5. **Performance**: Vectorized memory access and optimized thread layouts

This provides a complete template for integrating custom operators into the CUTLASS/CuTe DSL ecosystem.