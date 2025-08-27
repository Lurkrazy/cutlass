# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
AddThenPlus1 Custom Operator Example using CuTe DSL

This example demonstrates how to integrate a custom operator (AddThenPlus1) 
into the CuTe DSL lowering pipeline. The operator computes:

    out[i] = a[i] + b[i] + 1

The example provides:
1. Baseline computation: ref[i] = a[i] + b[i]
2. Custom operator computation: res[i] = AddThenPlus1(a[i], b[i])
3. Verification: res[i] == ref[i] + 1

To run this example:

.. code-block:: bash

    python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 1024 --N 512
    python examples/python/CuTeDSL/ampere/vector_add_then_plus1.py --M 1024 --N 1024 --benchmark
"""

import argparse
import time
from typing import Type
import sys
import os

# Add the parent directory to the path to import ops module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cuda.bindings.driver as cuda
import torch
import numpy as np

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from ops.add_then_plus1 import add_then_plus1


def run_add_then_plus1_test(
    M: int,
    N: int,
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    skip_ref_check: bool = False,
    benchmark: bool = False,
    warmup_iterations: int = 2,
    iterations: int = 100,
):
    """
    Run the AddThenPlus1 custom operator test
    
    Args:
        M: Number of rows
        N: Number of columns  
        dtype: Data type for computation
        skip_ref_check: Skip reference check if True
        benchmark: Run benchmark if True
        warmup_iterations: Number of warmup iterations for benchmark
        iterations: Number of iterations for benchmark
    """
    print(f"\n=== AddThenPlus1 Custom Operator Test ===")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Data type: {dtype}")
    print(f"Operation: out[i] = a[i] + b[i] + 1")

    torch_dtype = cutlass_torch.dtype(dtype)
    
    # Generate test data
    if dtype.is_integer:
        a = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
    else:
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

    c = torch.zeros_like(a)

    print(f"\nInput tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}")

    # Convert to CuTe tensors
    a_tensor = from_dlpack(a).mark_layout_dynamic()
    b_tensor = from_dlpack(b).mark_layout_dynamic()  
    c_tensor = from_dlpack(c).mark_layout_dynamic()

    print("\nCompiling AddThenPlus1 kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    if not skip_ref_check:
        print("\nExecuting custom AddThenPlus1 kernel...")
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        print("Verifying results...")
        
        # Compute reference: ref = a + b
        ref = a + b
        
        # Get result from custom operator: res = AddThenPlus1(a, b) = (a + b) + 1
        res = c
        
        # Verify: res should equal ref + 1
        expected = ref + 1.0
        
        # Check correctness
        torch.testing.assert_close(res, expected, atol=1e-6, rtol=1e-5)
        
        # Additional verification
        max_abs_err = torch.max(torch.abs(res - expected)).item()
        print(f"Max absolute error: {max_abs_err}")
        
        if max_abs_err < 1e-5:
            print("✓ PASS: AddThenPlus1 works correctly!")
            print(f"✓ Verified: res[i] == (a[i] + b[i]) + 1")
        else:
            print("✗ FAIL: Results do not match expected values")
            return False
            
        # Show sample results
        print(f"\nSample results (first 3x3 elements):")
        print(f"a[:3,:3] = \n{a[:3,:3]}")
        print(f"b[:3,:3] = \n{b[:3,:3]}")
        print(f"ref[:3,:3] = a + b = \n{ref[:3,:3]}")
        print(f"res[:3,:3] = AddThenPlus1(a,b) = \n{res[:3,:3]}")
        print(f"expected[:3,:3] = ref + 1 = \n{expected[:3,:3]}")

    if not benchmark:
        return True

    print(f"\n=== Benchmarking AddThenPlus1 ===")

    def generate_tensors():
        if dtype.is_integer:
            a = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
            b = torch.randint(
                0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype
            )
        else:
            a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
            b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

        c = torch.zeros_like(a)
        
        a_tensor = from_dlpack(a).mark_layout_dynamic()
        b_tensor = from_dlpack(b).mark_layout_dynamic()
        c_tensor = from_dlpack(c).mark_layout_dynamic()

        return testing.JitArguments(a_tensor, b_tensor, c_tensor)

    avg_time_us = testing.benchmark(
        compiled_func,
        workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # Print performance results
    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    
    # Calculate memory throughput (3 tensors: read A, read B, write C)
    total_elements = a.numel()
    bytes_per_element = dtype.width // 8
    total_bytes = 3 * total_elements * bytes_per_element  # A + B + C
    throughput_gb_s = total_bytes / (avg_time_us / 1e6) / 1e9
    
    print(f"Memory throughput: {throughput_gb_s:.2f} GB/s")
    print(f"Elements processed: {total_elements:,}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AddThenPlus1 Custom Operator Example for CuTe DSL"
    )
    parser.add_argument("--M", default=1024, type=int, help="Number of rows")
    parser.add_argument("--N", default=1024, type=int, help="Number of columns")
    parser.add_argument("--warmup_iterations", default=2, type=int, help="Warmup iterations for benchmark")
    parser.add_argument("--iterations", default=100, type=int, help="Benchmark iterations")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference check")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this example!")

    # Check if we're on Ampere or later (compute capability 8.0+)
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = device_props.major * 10 + device_props.minor
    
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    
    if compute_capability < 80:
        print("Warning: This example is optimized for Ampere GPUs (compute capability 8.0+)")

    success = run_add_then_plus1_test(
        M=args.M,
        N=args.N,
        dtype=cutlass.Float32,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )

    if success:
        print("\n🎉 All tests passed! AddThenPlus1 integration successful.")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()