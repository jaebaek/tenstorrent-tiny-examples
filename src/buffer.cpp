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

#include "buffer.h"

#include <functional>
#include <random>

#include "tt_metal/common/bfloat16.hpp"

namespace tiny {

template <>
Buffer<float>::Buffer(size_t number_of_elems, int seed) {
  tilized_ = false;
  all_zeros_ = false;

  buffer_.clear();
  buffer_.resize(number_of_elems);

  float rand_max = 2.0f;
  float offset = -1.0f;
  auto rand_elem = std::bind(std::uniform_real_distribution<float>(0, rand_max),
                             std::mt19937(seed));
  for (size_t i = 0; i < number_of_elems; ++i) {
    float elem = rand_elem() + offset;
    buffer_[i] = elem;
  }
}

template <>
Buffer<int>::Buffer(size_t number_of_elems, int seed) {
  tilized_ = false;
  all_zeros_ = false;

  buffer_.clear();
  buffer_.resize(number_of_elems);

  int rand_max = 200;
  int offset = 100;
  auto rand_elem = std::bind(std::uniform_real_distribution<int>(0, rand_max),
                             std::mt19937(seed));
  for (size_t i = 0; i < number_of_elems; ++i) {
    int elem = rand_elem() + offset;
    buffer_[i] = elem;
  }
}

template <>
Buffer<bfloat16>::Buffer(size_t number_of_elems, int seed) {
  tilized_ = false;

  buffer_.clear();
  buffer_.resize(number_of_elems);

  buffer_ = create_random_vector_of_bfloat16_native(
      /* num_bytes = */ number_of_elems * sizeof(bfloat16),
      /* rand_max_float = */ 2.0f,
      /* seed = */ seed,
      /* offset */ -1.0f);
  return;
}

} /* namespace tiny */
