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

#ifndef multicast_advanced_
#define multicast_advanced_

#include <memory>
#include <vector>

#include "blas_op.h"
#include "buffer.h"
#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace tiny {

/*
 * This example uses all Tensix cores. Each Tensix core reads a single tile and
 * sends it to other Tensix cores via multicasting. Finally, each Tensix core
 * write tiles that it read and received from other cores to output DRAM buffer.
 */
template <typename T>
class MulticastAdvanced : BLASOp {
 public:
  MulticastAdvanced(tt::tt_metal::Device* device)
      : device_(device) {}

  Result Run();

  /**
   * The number of tiles given by |input| must be the same as the number of
   * cores, because each Tensix core will read its own tile. The number of tiles
   * for the |output| must be the same as (the number of cores)^2, because each
   * core will keep all input tiles to its output DRAM slot.
   */
  void SetBuffers(std::shared_ptr<Buffer<T>> input,
                  std::shared_ptr<Buffer<T>> output) {
    auto core_grid = device_->compute_with_storage_grid_size();
    uint32_t num_cores = core_grid.x * core_grid.y;
    assert(input->GetSizeInBytes() == num_cores * tiny::SingleTileSize<T>());
    assert(output->GetSizeInBytes() ==
           num_cores * num_cores * tiny::SingleTileSize<T>());

    input_ = input;
    output_ = output;
  }

 private:
  std::shared_ptr<Buffer<T>> input_;
  std::shared_ptr<Buffer<T>> output_;
  tt::tt_metal::Device* device_;
};

} /* namespace tiny */

#endif /* ifndef multicast_advanced_ */
