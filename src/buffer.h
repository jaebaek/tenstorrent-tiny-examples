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

#ifndef buffer_h_
#define buffer_h_

#include <vector>

#include "utils.h"

namespace tiny {

/* Support only bfloat16, float, int. */
template <typename T>
class Buffer {
 public:
  Buffer() : tilized_(false), all_zeros_(false) {}

  /* Set all |number_of_elems| elements as 0. */
  Buffer(size_t number_of_elems) : tilized_(false) {
    buffer_.clear();
    buffer_.resize(number_of_elems, 0);
    all_zeros_ = true;
  }

  /* Set all |number_of_elems| elements as |value|. */
  Buffer(size_t number_of_elems, const T& value)
      : tilized_(false), all_zeros_(false) {
    buffer_.clear();
    buffer_.resize(number_of_elems, value);
  }

  /* Set all |number_of_elems| elements as random values. */
  Buffer(size_t number_of_elems, int seed);

  size_t GetNumberOfElements() const { return buffer_.size(); }

  size_t GetSizeInBytes() const { return buffer_.size() * sizeof(T); }

  std::vector<T>& GetVector() { return buffer_; }

  void Tilize(uint32_t width, uint32_t height) {
    // Return if it is already tilized.
    if (tilized_) return;

    TilizeForTTDevice<T>(buffer_, width, height);
    tilized_ = true;
  }

  void Untilize(uint32_t width, uint32_t height) {
    // Return if it is already untilized.
    if (!tilized_) return;

    UnTilizeForTTDevice<T>(buffer_, width, height);
    tilized_ = false;
  }

  bool IsTilized() const { return tilized_; }

  bool AllZeros() const { return all_zeros_; }

  ~Buffer() {}

 private:
  std::vector<T> buffer_;
  bool tilized_;
  bool all_zeros_;
};

} /* namespace tiny */

#endif /* ifndef buffer_h_ */
