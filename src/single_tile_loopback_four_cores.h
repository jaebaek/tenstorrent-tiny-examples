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

#ifndef single_tile_loopback_four_cores_
#define single_tile_loopback_four_cores_

#include <cassert>
#include <memory>
#include <vector>

#include "blas_op.h"
#include "buffer.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "utils.h"

namespace tiny {

/**
 * This example will copy the |input| tile to the |output| buffer 4 times. The
 * expected result is that the |output| buffer has 4 copies of |input| tiles.
 * Therefore the size of the |output| buffer must be 4 times of the |input|
 * tile. The copy will be done by 4 Tensix cores. We write this example to
 * compare the normal data copy (via DRAM to L1) and the multicast. See the
 * SimpleMulticast example.
 */
template <typename T>
class SingleTileLoopbackFourCores : BLASOp {
 public:
  Result Run();

  void SetBuffers(std::shared_ptr<Buffer<T>> input,
                  std::shared_ptr<Buffer<T>> output) {
    assert(input->GetSizeInBytes() == tiny::SingleTileSize<T>());
    assert(output->GetSizeInBytes() == 4 * tiny::SingleTileSize<T>());

    input_ = input;
    output_ = output;
  }

 private:
  std::shared_ptr<Buffer<T>> input_;
  std::shared_ptr<Buffer<T>> output_;
};

} /* namespace tiny */

#endif /* ifndef single_tile_loopback_four_cores_ */
