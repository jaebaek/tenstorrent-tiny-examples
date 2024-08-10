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

#define TINY_DEBUG 0

#if TINY_DEBUG
#define LOG(X) DPRINT_DATA1(X)
#else
#define LOG(X)
#endif

static inline SliceRange hw_all() {
  return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
}

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t input_dram_addr = get_arg_val<uint32_t>(1);
  uint32_t output_dram_addr = get_arg_val<uint32_t>(2);

  const uint32_t tile_size_in_bytes = get_tile_size(tt::CB::c_in0);
  const DataFormat format = get_dataformat(tt::CB::c_in0);
  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input = {
      .bank_base_address = input_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

  /**
   * Note that we must set `.bank_base_address` field the same as the address
   * returned by the host (CPU) API `CreateBuffer(..)`. If you pass the one
   * added by some numbers like
   * `output_dram_addr + core_id * tile_size_in_bytes`,
   * it will not send the data to the output buffer.
   *
   * In order to send the data to the location of
   * `core_id * tile_size_in_bytes`, you must set `id` parameter of
   * `noc_async_write_tile(..)` function.
   */
  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_output = {
      .bank_base_address = output_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

#if TINY_DEBUG
  LOG(DPRINT << "input_dram_addr: " << input_dram_addr << ENDL());
  LOG(DPRINT << "output_dram_addr: "
             << output_dram_addr + core_id * tile_size_in_bytes << ENDL());
#endif

  // Read a single tile from DRAM |input_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
  noc_async_read_tile(0, bank_for_input, L1_write_addr_in0);
  noc_async_read_barrier();

#if TINY_DEBUG
  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0);
  LOG(DPRINT << *ptr << ENDL());
  ptr = reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0 + 4);
  LOG(DPRINT << *ptr << ENDL());
#endif

#if TINY_DEBUG
  LOG(DPRINT << TSLICE(tt::CB::c_in0, 0, hw_all()) << ENDL());
#endif

  // Write a single tile from circular buffer in0 to DRAM |output_dram_addr|.
  uint32_t L1_read_addr_in0 = get_read_ptr(tt::CB::c_in0);
  noc_async_write_tile(core_id, bank_for_output, L1_read_addr_in0);
  noc_async_write_barrier();
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);
}
