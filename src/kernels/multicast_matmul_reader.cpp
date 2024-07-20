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

const uint32_t PHYSICAL_CORE_Y[] = {1, 3, 4, 5, 7, 8, 9, 10};

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t core_grid_x = get_arg_val<uint32_t>(1);
  uint32_t core_grid_y = get_arg_val<uint32_t>(2);
  uint32_t input0_dram_addr = get_arg_val<uint32_t>(3);
  uint32_t input1_dram_addr = get_arg_val<uint32_t>(4);
  uint32_t receiver_sema_addr = get_arg_val<uint32_t>(5);
  uint32_t sender_sema_addr = get_arg_val<uint32_t>(6);

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input0 = {
      .bank_base_address = input0_dram_addr,
      .page_size = get_tile_size(tt::CB::c_in0),
      .data_format = get_dataformat(tt::CB::c_in0)};

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input1 = {
      .bank_base_address = input1_dram_addr,
      .page_size = get_tile_size(tt::CB::c_in1),
      .data_format = get_dataformat(tt::CB::c_in1)};

  // Read a single tile from DRAM |input0_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
  bank_for_input0.noc_async_read_tile(core_id, L1_write_addr_in0);
  noc_async_read_barrier();
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);

  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0);
  LOG(DPRINT << "[READER] dram -> cb0: " << *(ptr) << ENDL());
  ptr = reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0 + 4);
  LOG(DPRINT << "[READER] dram -> cb0: " << *(ptr) << ENDL());

  uint32_t number_of_cores = core_grid_x * core_grid_y;

  // ---- Multi-casting start ----
  // Based on multi-casting,
  //  1. Receive i-th tile of input1 matrix from i-th Tensix core.
  //  2. Send |core_id|-th tile of input1 matrix to all other Tensix cores.

  LOG(DPRINT << "[READER] Multicast start" << ENDL());

  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  for (uint32_t i = 0; i < core_id; ++i) {
    noc_semaphore_set(receiver_sema_addr_ptr, 0);

    uint32_t sender_noc_x = i % core_grid_x + 1;
    uint32_t sender_noc_y = PHYSICAL_CORE_Y[i / core_grid_x];
    uint64_t sender_sema_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_sema_addr);
    LOG(DPRINT << "[READER] sema inc " << core_id << ", " << i << ENDL());
    LOG(DPRINT << "[READER] sender_sema_noc_addr " << sender_sema_noc_addr
               << ENDL());
    noc_semaphore_inc(sender_sema_noc_addr, 1);

    // We use c_in2 for the tile received from other Tensix cores.
    cb_reserve_back(tt::CB::c_in2, /* number of tiles */ 1);
    LOG(DPRINT << "[READER] wait " << core_id << ", " << i << ENDL());
    LOG(DPRINT << "[READER] receiver_sema_addr_ptr " << receiver_sema_addr
               << ", " << i << ENDL());

    noc_semaphore_wait(receiver_sema_addr_ptr, 1);

#if TINY_DEBUG  // Print first float from CB2 for debugging.
    uint32_t L1_write_addr_cb2 = get_write_ptr(tt::CB::c_in2);
    volatile tt_l1_ptr float* ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_cb2);
    LOG(DPRINT << "[READER] receive cb2: " << *(ptr_first_float) << ENDL());
    ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_cb2 + 4);
    LOG(DPRINT << "[READER] receive cb2: " << *(ptr_first_float) << ENDL());
#endif

    LOG(DPRINT << "[READER] done " << core_id << ", " << i << ENDL());
    cb_push_back(tt::CB::c_in2, /* number of tiles */ 1);
  }

  LOG(DPRINT << "[READER] sender sema wait " << sender_sema_addr << ENDL());

  volatile tt_l1_ptr uint32_t* sender_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sema_addr);
  noc_semaphore_wait(sender_sema_addr_ptr, number_of_cores - 1);
  noc_semaphore_set(sender_sema_addr_ptr, 0);

  // We use c_in2 for the tile sent to other Tensix cores.
  cb_reserve_back(tt::CB::c_in2, /* number of tiles */ 1);
  uint32_t L1_write_addr_in2 = get_write_ptr(tt::CB::c_in2);

  // Read a single tile from DRAM |input1_dram_addr| to circular buffer in1.
  cb_reserve_back(tt::CB::c_in1, /* number of tiles */ 1);
  uint32_t L1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
  bank_for_input1.noc_async_read_tile(core_id, L1_write_addr_in1);

  // Barrier for the read from |input1_dram_addr| to |L1_write_addr_in1|.
  noc_async_read_barrier();

  uint64_t multicast_dst_noc_addr =
      get_noc_multicast_addr(core_grid_x, 10, 1, 1, L1_write_addr_in2);
  const uint32_t tile_size_in_bytes = get_tile_size(tt::CB::c_in1);
  // Based on mcast example provided by Tenstorrent, the number of destinations
  // must not include source, since we are NOT really doing a local copy.
  noc_async_write_multicast(L1_write_addr_in1, multicast_dst_noc_addr,
                            tile_size_in_bytes, number_of_cores - 1);

  LOG(DPRINT << "[READER] send sema release" << ENDL());

  *(receiver_sema_addr_ptr) = 1;  // Unlock semaphores of all receivers.
  uint64_t noc_addr =
      get_noc_multicast_addr(core_grid_x, 10, 1, 1, receiver_sema_addr);
  // Based on mcast example provided by Tenstorrent, the number of destinations
  // must not include source, since we are NOT really doing a local copy.
  noc_semaphore_set_multicast(receiver_sema_addr, noc_addr,
                              number_of_cores - 1);
  noc_async_write_barrier();

  // Note that we do not need both `noc_async_write_barrier()` and
  // `noc_async_writes_flushed()` here. The barrier helps us to guarantee that
  // the multi-cast write is done from receiver side and let it safely start
  // the compute. Since the receiver waits for the semaphore, if it passes the
  // wait line, it means the multi-cast write was already done.

#if TINY_DEBUG  // Print first float from CB1 for debugging.
  volatile tt_l1_ptr float* ptr_first_float_from_input1 =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1);
  LOG(DPRINT << "[READER] send cb1: " << *(ptr_first_float_from_input1)
             << ENDL());
  ptr_first_float_from_input1 =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1 + 4);
  LOG(DPRINT << "[READER] send cb1: " << *(ptr_first_float_from_input1)
             << ENDL());
#endif

  LOG(DPRINT << "[READER] CB push back" << ENDL());

  cb_push_back(tt::CB::c_in2, /* number of tiles */ 1);
  cb_push_back(tt::CB::c_in1, /* number of tiles */ 1);

  for (uint32_t i = core_id + 1; i < number_of_cores; ++i) {
    noc_semaphore_set(receiver_sema_addr_ptr, 0);

    uint32_t sender_noc_x = i % core_grid_x + 1;
    uint32_t sender_noc_y = PHYSICAL_CORE_Y[i / core_grid_x];
    uint64_t sender_sema_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_sema_addr);
    LOG(DPRINT << "[READER] sema inc " << core_id << ", " << i << ENDL());
    LOG(DPRINT << "[READER] sender_sema_noc_addr " << sender_sema_noc_addr
               << ENDL());
    noc_semaphore_inc(sender_sema_noc_addr, 1);

    // We use c_in2 for the tile received from other Tensix cores.
    cb_reserve_back(tt::CB::c_in2, /* number of tiles */ 1);
    LOG(DPRINT << "[READER] wait " << core_id << ", " << i << ENDL());
    LOG(DPRINT << "[READER] receiver_sema_addr_ptr " << receiver_sema_addr
               << ", " << i << ENDL());

    noc_semaphore_wait(receiver_sema_addr_ptr, 1);

#if TINY_DEBUG  // Print first float from CB2 for debugging.
    uint32_t L1_write_addr_cb2 = get_write_ptr(tt::CB::c_in2);
    volatile tt_l1_ptr float* ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_cb2);
    LOG(DPRINT << "[READER] receive cb2: " << *(ptr_first_float) << ENDL());
    ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_cb2 + 4);
    LOG(DPRINT << "[READER] receive cb2: " << *(ptr_first_float) << ENDL());
#endif

    LOG(DPRINT << "[READER] done " << core_id << ", " << i << ENDL());
    cb_push_back(tt::CB::c_in2, /* number of tiles */ 1);
  }
  // ---- Multi-casting end ----
}
