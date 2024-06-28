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

#ifndef multicast_matmul_
#define multicast_matmul_

#include <memory>
#include <vector>

#include "blas_op.h"
#include "buffer.h"
#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace tiny {

/*
 * This class conducts matrix multiplication for two matrices:
 * - A whose dimension is K by TileWidth()
 * - B whose dimension is TileWidth() by K
 *
 * where K is `|number of Tensix cores| * TileWidth()`.
 *
 * i-th Tensix core owns the computation for i-th row of the output matrix.
 * i-th Tensix core reads i-th tile row of A and i-th tile column of B and
 * multi-casts i-th tile column of B to all other Tensix cores. In addition,
 * it receives all tiles of B other than i-th tile column of B from other
 * Tensix cores.
 */
template <typename T>
class MulticastMatrixMultiplication : BLASOp {
 public:
  MulticastMatrixMultiplication() : device_(nullptr) {}

  ~MulticastMatrixMultiplication() {
    if (device_ != nullptr) {
      tt::tt_metal::CloseDevice(device_.get());
    }
  }

  Result Run();

  void SetBuffers(std::shared_ptr<Buffer<T>> input0,
                  std::shared_ptr<Buffer<T>> input1,
                  std::shared_ptr<Buffer<T>> output) {
    auto core_grid = GetOrCreateDevice()->compute_with_storage_grid_size();
    uint32_t num_cores = core_grid.x * core_grid.y;
    assert(input0->GetSizeInBytes() == num_cores * tiny::SingleTileSize<T>());
    assert(input1->GetSizeInBytes() == num_cores * tiny::SingleTileSize<T>());
    assert(output->GetSizeInBytes() ==
           num_cores * num_cores * tiny::SingleTileSize<T>());

    inputs_[0] = input0;
    inputs_[1] = input1;
    output_ = output;
  }

  std::shared_ptr<tt::tt_metal::Device> GetOrCreateDevice() {
    if (device_ == nullptr) {
      constexpr int device_id = 0;
      device_ = std::shared_ptr<tt::tt_metal::Device>(
          tt::tt_metal::CreateDevice(device_id));
    }
    return device_;
  }

 private:
  std::shared_ptr<Buffer<T>> inputs_[2];
  std::shared_ptr<Buffer<T>> output_;
  std::shared_ptr<tt::tt_metal::Device> device_;
};

} /* namespace tiny */

#endif /* ifndef multicast_matmul_ */
