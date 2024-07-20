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
#define LOG(X) DPRINT_DATA1(X)
#else
#define LOG(X)
#endif

void kernel_main() {
  uint32_t input_dram_addr = get_arg_val<uint32_t>(0);
  uint32_t receiver_sema_addr = get_arg_val<uint32_t>(1);

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input = {
      .bank_base_address = input_dram_addr,
      .page_size = get_tile_size(tt::CB::c_in0),
      .data_format = get_dataformat(tt::CB::c_in0)};

  // Read a single tile from DRAM |input_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
  bank_for_input.noc_async_read_tile(0, L1_write_addr_in0);
  noc_async_read_barrier();

  // Send CB::c_in0 tile to receiver's CB::c_in1.
  cb_reserve_back(tt::CB::c_in1, /* number of tiles */ 1);
  uint32_t L1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);

#if TINY_DEBUG  // Print first float from CB1 for debugging.
  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0);
  LOG(DPRINT << "[READER] send cb0: " << *(ptr) << ENDL());
#endif

  uint64_t multicast_dst_noc_addr =
      get_noc_multicast_addr(1, 5, 1, 3, L1_write_addr_in1);
  const uint32_t tile_size_in_bytes = get_tile_size(tt::CB::c_in1);
  noc_async_write_multicast(L1_write_addr_in0, multicast_dst_noc_addr,
                            tile_size_in_bytes, 3 /* to {1, 3 .. 5}*/);

  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  *(receiver_sema_addr_ptr) = 1;  // Unlock semaphores of all receivers.
  uint64_t noc_addr = get_noc_multicast_addr(1, 5, 1, 3, receiver_sema_addr);
  noc_semaphore_set_multicast(receiver_sema_addr, noc_addr,
                              3 /* to {1, 3 .. 5}*/);
  noc_async_write_barrier();

  cb_push_back(tt::CB::c_in1, /* number of tiles */ 1);
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);
}
