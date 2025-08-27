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
AddThenPlus1 operator implementation for CuTe DSL

This module implements a custom operator that computes: 
out[i] = a[i] + b[i] + 1

The operator is designed to integrate with the CuTe DSL lowering pipeline
and demonstrates how to create custom epilogue functors for CUTLASS operations.
"""

import cutlass
import cutlass.cute as cute


@cute.kernel
def add_then_plus1_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    """
    Custom kernel implementing AddThenPlus1 operation: C[i] = A[i] + B[i] + 1
    
    This kernel demonstrates how to implement a custom epilogue operation
    that adds two vectors and then adds 1 to each element.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]  # (TileM,TileN)
    blkC = gC[blk_coord]  # (TileM,TileN)
    blkCrd = cC[blk_coord]  # (TileM, TileN)

    # Print tensor information at compile time
    print(f"[DSL INFO] AddThenPlus1 Kernel - Sliced Tensors per thread block:")
    print(f"[DSL INFO]   blkA = {blkA.type}")
    print(f"[DSL INFO]   blkB = {blkB.type}")
    print(f"[DSL INFO]   blkC = {blkC.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # declare the atoms which will be used later for memory copy
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # allocate fragments for gmem->rmem
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # Create predicate fragment for bounds checking
    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    print(f"[DSL INFO] AddThenPlus1 Kernel - Sliced Tensors per thread:")
    print(f"[DSL INFO]   thrA = {thrA.type}")
    print(f"[DSL INFO]   thrB = {thrB.type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    # Generate predicate mask for bounds checking
    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    ##########################################################
    # Move data to register address space
    ##########################################################

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    # Perform the AddThenPlus1 operation: (a + b) + 1
    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    a_values = frgA.load()
    b_values = frgB.load()
    
    # Custom AddThenPlus1 operation
    result = a_values + b_values + 1.0

    # Save the results back to registers
    frgC.store(result)

    # Copy the results back to global memory
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def add_then_plus1(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    """
    Compiles and executes the AddThenPlus1 operation
    
    Args:
        mA: Input tensor A
        mB: Input tensor B  
        mC: Output tensor C
        copy_bits: Number of bits for vectorized copies (default: 128)
        
    Returns:
        Compiled function that can be called with tensor arguments
    """
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] AddThenPlus1 - Input Tensors:")
    print(f"[DSL INFO]   mA = {mA.type}")
    print(f"[DSL INFO]   mB = {mB.type}")
    print(f"[DSL INFO]   mC = {mC.type}")

    print(f"[DSL INFO] AddThenPlus1 - Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    
    print(f"[DSL INFO] AddThenPlus1 - Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    print(f"[DSL INFO]   coord tensor = {cC.type}")

    add_then_plus1_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def create_add_then_plus1_op():
    """
    Factory function to create an AddThenPlus1 operator instance
    
    Returns:
        A function that implements the AddThenPlus1 operation
    """
    return add_then_plus1