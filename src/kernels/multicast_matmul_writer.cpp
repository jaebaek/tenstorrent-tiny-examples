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

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t number_of_cores = get_arg_val<uint32_t>(1);
  uint32_t output_dram_addr = get_arg_val<uint32_t>(2);

  const uint32_t tile_size = get_tile_size(tt::CB::c_out0);
  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_output = {
      .bank_base_address = output_dram_addr + number_of_cores * core_id * tile_size,
      .page_size = tile_size,
      .data_format = get_dataformat(tt::CB::c_out0)};

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    cb_wait_front(tt::CB::c_out1, /* number of tiles */ 1);
    uint32_t addr = get_read_ptr(tt::CB::c_out1);
    cb_pop_front(tt::CB::c_out1, /* number of tiles */ 1);

    cb_wait_front(tt::CB::c_out2, /* number of tiles */ 1);
    addr = get_read_ptr(tt::CB::c_out2);
    cb_pop_front(tt::CB::c_out2, /* number of tiles */ 1);

    cb_wait_front(tt::CB::c_out0, /* number of tiles */ 1);
    uint32_t L1_read_addr_out = get_read_ptr(tt::CB::c_out0);
    noc_async_write_tile(i, bank_for_output, L1_read_addr_out);
    noc_async_write_barrier();
    cb_pop_front(tt::CB::c_out0, /* number of tiles */ 1);
  }
}
