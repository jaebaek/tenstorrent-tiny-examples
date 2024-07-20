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
#include "debug/dprint.h"

#define TINY_DEBUG 1

#if TINY_DEBUG
#define LOG(X) DPRINT_DATA0(X)
#else
#define LOG(X)
#endif

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t output_dram_addr = get_arg_val<uint32_t>(1);

  const uint32_t tile_size = get_tile_size(tt::CB::c_out0);

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_output = {
      .bank_base_address = output_dram_addr + core_id * tile_size,
      .page_size = tile_size,
      .data_format = get_dataformat(tt::CB::c_out0)};

  LOG(DPRINT << "[WRITER] wait" << ENDL());

  cb_wait_front(tt::CB::c_out0, /* number of tiles */ 1);
  uint32_t L1_read_addr_out = get_read_ptr(tt::CB::c_out0);
#if TINY_DEBUG  // Print first float from CB1 for debugging.
  volatile tt_l1_ptr float* ptr_first_float =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_read_addr_out);
  LOG(DPRINT << "[WRITER] receive c_out0: " << *(ptr_first_float) << ENDL());
#endif
  bank_for_output.noc_async_write_tile(0, L1_read_addr_out);
  noc_async_write_barrier();
  cb_pop_front(tt::CB::c_out0, /* number of tiles */ 1);
}
