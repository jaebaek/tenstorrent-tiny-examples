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

#ifndef conv_h_
#define conv_h_

#include <cassert>
#include <memory>

#include "blas_op.h"
#include "buffer.h"
#include "utils.h"

namespace tiny {

/**
 * Class for simple convolution operation example.
 *
 * Input dimension: (64, 96, 32)
 *   height = 64
 *   width = 96
 *   number of channels = 32
 *
 * Weight dimension: (4, 4, 32, 128)
 *   height = 4
 *   width = 4
 *   number of input channels = 32 (the same as the input channels)
 *   number of output channels = 128 (the same as the output channels)
 *
 * Slide: (1, 1) - one by one slide for both horizontal and virtical directions.
 * Padding: (2, 2)
 *
 * Output dimension: (64, 96, 128)
 *   output_h = (input_h + 2 * padding_h - weight_h) / slide_h
 *   output_w = (input_w + 2 * padding_w - weight_w) / slide_w
 *
 * For both Conv and CpuConv, we assume that the given input buffer has an
 * order of elements based on the following rule:
 *  - The first row of the first channel matrix is placed first.
 *  - The second row of the second channel matrix is placed second.
 *  - ...
 *
 * The given weight buffer has the same order of elements i.e., the first row
 * of the first first channel matrix is placed first.
 */
template <typename T>
class Conv : BLASOp {
 public:
  Result Run();

  void SetBuffers(std::shared_ptr<Buffer<T>> input,
                  std::shared_ptr<Buffer<T>> weight,
                  std::shared_ptr<Buffer<T>> output) {
    input_ = input;
    weight_ = weight;
    output_ = output;
    CheckDimension();
  }

 protected:
  std::shared_ptr<Buffer<T>> input_;
  std::shared_ptr<Buffer<T>> weight_;
  std::shared_ptr<Buffer<T>> output_;

 private:
  void CheckDimension();
};

template <typename T>
class CpuConv : public Conv<T> {
 public:
  Result Run();
};

} /* namespace tiny */

int run(int argc, const char* argv[]);

#endif /* ifndef conv_h_ */
