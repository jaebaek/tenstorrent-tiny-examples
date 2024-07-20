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

#ifndef simple_multicast_
#define simple_multicast_

#include <cassert>
#include <memory>

#include "blas_op.h"
#include "buffer.h"
#include "utils.h"

namespace tiny {

/**
 * This class will use 4 Tensix cores. The first core will be the sender of the
 * multicast. It will send a tile to 3 receiver cores.
 */
template <typename T>
class SimpleMulticast : BLASOp {
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

#endif /* ifndef simple_multicast_ */
