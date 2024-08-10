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

#ifndef single_tile_matmul_
#define single_tile_matmul_

#include <cassert>
#include <memory>
#include <vector>

#include "blas_op.h"
#include "buffer.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "utils.h"

namespace tiny {

template <typename T>
class SingleTileMatrixMultiplication : BLASOp {
 public:
  Result Run();

  void SetBuffers(std::shared_ptr<Buffer<T>> input0,
                  std::shared_ptr<Buffer<T>> input1,
                  std::shared_ptr<Buffer<T>> output) {
    // Make sure the sizes of all buffers are a single tile size.
    assert(input0->GetSizeInBytes() == tiny::SingleTileSize<T>());
    assert(input1->GetSizeInBytes() == tiny::SingleTileSize<T>());
    assert(output->GetSizeInBytes() == tiny::SingleTileSize<T>());

    inputs_[0] = input0;
    inputs_[1] = input1;
    output_ = output;
  }

 private:
  std::shared_ptr<Buffer<T>> inputs_[2];
  std::shared_ptr<Buffer<T>> output_;
};

} /* namespace tiny */

#endif /* ifndef single_tile_matmul_ */