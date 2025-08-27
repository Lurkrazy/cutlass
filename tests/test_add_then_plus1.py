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
Pytest tests for AddThenPlus1 custom operator

These tests verify the correctness of the AddThenPlus1 operator
which computes out[i] = a[i] + b[i] + 1
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add the examples directory to the path
examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'python', 'CuTeDSL')
sys.path.append(examples_dir)

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# Import the custom operator
from ops.add_then_plus1 import add_then_plus1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAddThenPlus1:
    """Test suite for AddThenPlus1 custom operator"""

    def test_add_then_plus1_small(self):
        """Test AddThenPlus1 with small tensor (1024 elements)"""
        n = 1024
        a = np.arange(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        
        # Convert to PyTorch tensors on GPU
        a_torch = torch.from_numpy(a).cuda()
        b_torch = torch.from_numpy(b).cuda()
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # Verify results
        ref = a_torch + b_torch  # Reference: a + b
        expected = ref + 1.0     # Expected: (a + b) + 1
        
        torch.testing.assert_close(c_torch, expected, atol=1e-6)

    def test_add_then_plus1_2d(self):
        """Test AddThenPlus1 with 2D tensor"""
        M, N = 32, 64
        
        # Generate test data
        a_torch = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b_torch = torch.randn(M, N, device="cuda", dtype=torch.float32)
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # Verify results
        ref = a_torch + b_torch  # Reference: a + b
        expected = ref + 1.0     # Expected: (a + b) + 1
        
        torch.testing.assert_close(c_torch, expected, atol=1e-6)

    def test_add_then_plus1_specific_values(self):
        """Test AddThenPlus1 with specific known values"""
        # Create specific test case where we know the expected result
        a_values = [1.0, 2.0, 3.0, 4.0]
        b_values = [0.5, 1.5, 2.5, 3.5]
        expected_values = [2.5, 4.5, 6.5, 8.5]  # (a + b) + 1
        
        a_torch = torch.tensor(a_values, device="cuda", dtype=torch.float32)
        b_torch = torch.tensor(b_values, device="cuda", dtype=torch.float32)
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # Verify results
        expected_torch = torch.tensor(expected_values, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(c_torch, expected_torch, atol=1e-6)

    def test_add_then_plus1_zeros(self):
        """Test AddThenPlus1 with zero inputs"""
        M, N = 16, 32
        
        # Zero inputs
        a_torch = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        b_torch = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # With zero inputs, result should be all 1.0
        expected = torch.ones(M, N, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(c_torch, expected, atol=1e-6)

    def test_add_then_plus1_negative_values(self):
        """Test AddThenPlus1 with negative values"""
        M, N = 8, 16
        
        # Negative values
        a_torch = torch.full((M, N), -2.0, device="cuda", dtype=torch.float32)
        b_torch = torch.full((M, N), -3.0, device="cuda", dtype=torch.float32)
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # Expected: (-2) + (-3) + 1 = -4
        expected = torch.full((M, N), -4.0, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(c_torch, expected, atol=1e-6)

    @pytest.mark.parametrize("M,N", [(64, 128), (128, 256), (256, 512)])
    def test_add_then_plus1_various_sizes(self, M, N):
        """Test AddThenPlus1 with various tensor sizes"""
        # Generate random test data
        a_torch = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b_torch = torch.randn(M, N, device="cuda", dtype=torch.float32)
        c_torch = torch.zeros_like(a_torch)
        
        # Convert to CuTe tensors
        a_tensor = from_dlpack(a_torch).mark_layout_dynamic()
        b_tensor = from_dlpack(b_torch).mark_layout_dynamic()
        c_tensor = from_dlpack(c_torch).mark_layout_dynamic()
        
        # Compile and execute
        compiled_func = cute.compile(add_then_plus1, a_tensor, b_tensor, c_tensor)
        compiled_func(a_tensor, b_tensor, c_tensor)
        
        # Verify results
        ref = a_torch + b_torch  # Reference: a + b
        expected = ref + 1.0     # Expected: (a + b) + 1
        
        torch.testing.assert_close(c_torch, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])