/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing addition with plus one operation used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies addition with plus one operation to an array of elements.
///
/// D = (alpha * accumulator + beta * source) + 1
/// 
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation.
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_
>
class AddThenPlus1 {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params 
  {
    ElementCompute alpha;                         ///< scales accumulators
    ElementCompute beta;                          ///< scales source tensor
    ElementCompute const *alpha_ptr;              ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;               ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute const* const* alpha_ptr_array; ///< array of pointers to accumulator scalar per group/batch
    ElementCompute const* const* beta_ptr_array;  ///< array of pointers to source scalar per group/batch

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr),
      alpha_ptr_array(nullptr),
      beta_ptr_array(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta = ElementCompute(0)
    ): alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr),
       alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr = nullptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr),
       alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {

    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  AddThenPlus1(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::OnlyAlphaPerChannelScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes: D = (alpha * accumulator + beta * source) + 1
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform the operation: (alpha * accumulator + beta * source) + 1
    FragmentCompute intermediate;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      intermediate[i] = alpha_ * converted_accumulator[i] + beta_ * converted_source[i] + ElementCompute(1);
    }

    // Convert to output numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes: D = (alpha * accumulator) + 1 (when source is not needed)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert accumulator to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform the operation: (alpha * accumulator) + 1
    FragmentCompute intermediate;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      intermediate[i] = alpha_ * converted_accumulator[i] + ElementCompute(1);
    }

    // Convert to output numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////