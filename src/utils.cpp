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

#include "utils.h"

namespace tiny {

std::vector<uint32_t> GetPhysicalCoreCoord(tt::tt_metal::Device* device,
                                           CoreCoord core_grid) {
  std::vector<uint32_t> physical_core_coord_info;
  for (uint32_t x = 0; x < core_grid.x; ++x) {
    CoreCoord core = {x, 0};
    auto core_physical = device->worker_core_from_logical_core(core);
    physical_core_coord_info.push_back(core_physical.x);
  }
  for (uint32_t y = 0; y < core_grid.y; ++y) {
    CoreCoord core = {0, y};
    auto core_physical = device->worker_core_from_logical_core(core);
    physical_core_coord_info.push_back(core_physical.y);
  }
  return std::move(physical_core_coord_info);
}

} /* namespace tiny */
