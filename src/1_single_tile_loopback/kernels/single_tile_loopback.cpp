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
  uint32_t input_dram_addr = get_arg_val<uint32_t>(0);
  uint32_t output_dram_addr = get_arg_val<uint32_t>(1);

  const uint32_t tile_size_in_bytes = get_tile_size(tt::CB::c_in0);
  const DataFormat format = get_dataformat(tt::CB::c_in0);
  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input = {
      .bank_base_address = input_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_output = {
      .bank_base_address = output_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

  // Read a single tile from DRAM |input_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
  bank_for_input.noc_async_read_tile(0, L1_write_addr_in0);
  noc_async_read_barrier();

  // Write a single tile from circular buffer in0 to DRAM |output_dram_addr|.
  uint32_t L1_read_addr_in0 = get_read_ptr(tt::CB::c_in0);
  bank_for_output.noc_async_write_tile(0, L1_read_addr_in0);
  noc_async_write_barrier();
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);
}
