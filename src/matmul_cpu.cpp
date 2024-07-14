// Copyright (c) 2024 Jaebaek Seo.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "matmul_cpu.h"

#include <cassert>
#include <tuple>

#include "buffer.h"
#include "tt_metal/common/bfloat16.hpp"

namespace {

static constexpr uint32_t kTileWidthCPU = 8;
static constexpr uint32_t kTileHeightCPU = 8;

/*
 * The following code shows the idea of "tiling".
 */
template <typename T>
tiny::Result _Run(std::shared_ptr<tiny::Buffer<T>> input0,
                  std::shared_ptr<tiny::Buffer<T>> input1,
                  std::shared_ptr<tiny::Buffer<T>> output, uint32_t m_,
                  uint32_t k_, uint32_t n_) {
  auto& input_vec0 = input0->GetVector();
  auto& input_vec1 = input1->GetVector();
  auto& output_vec = output->GetVector();

  for (uint32_t i = 0; i < m_; i += kTileHeightCPU) {
    for (uint32_t j = 0; j < n_; j += kTileWidthCPU) {
      for (uint32_t k = 0; k < k_; ++k) {
        // Multiplication for a single tile.
        for (uint32_t ti = i; ti < i + kTileHeightCPU; ++ti) {
          for (uint32_t tj = j; tj < j + kTileWidthCPU; ++tj) {
            uint32_t index0 = ti * k_ + k;
            uint32_t index1 = k * k_ + tj;
            output_vec[ti * n_ + tj] += input_vec0[index0] * input_vec1[index1];
          }
        }
      }
    }
  }

  return tiny::Result::kSuccess;
}

} /* namespace */

namespace tiny {

template <>
Result CPUMatrixMultiplication<bfloat16>::Run() {
  assert(!inputs_[0]->IsTilized());
  assert(!inputs_[1]->IsTilized());

  for (uint32_t i = 0; i < m_; ++i) {
    for (uint32_t j = 0; j < n_; ++j) {
      float element = 0.0f;
      for (uint32_t k = 0; k < k_; ++k) {
        uint32_t index0 = i * k_ + k;
        uint32_t index1 = k * k_ + j;
        bfloat16 value(inputs_[0]->GetVector()[index0].to_float() *
                       inputs_[1]->GetVector()[index1].to_float());
        element += value.to_float();
      }
      output_->GetVector()[i * n_ + j] = bfloat16(element);
    }
  }

  return Result::kSuccess;
}

template <>
Result CPUMatrixMultiplication<float>::Run() {
  return _Run(inputs_[0], inputs_[1], output_, m_, k_, n_);
}

} /* namespace tiny */
