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

constexpr uint32_t core_grid_x = get_compile_time_arg_val(0);
constexpr uint32_t core_grid_y = get_compile_time_arg_val(1);
constexpr uint32_t number_of_cores = core_grid_x * core_grid_y;
uint32_t PHYSICAL_CORES_X[core_grid_x];
uint32_t PHYSICAL_CORES_Y[core_grid_y];

static void init_physical_cores(const uint32_t start_arg_index) {
  for (uint32_t i = 0; i < core_grid_x; ++i) {
    PHYSICAL_CORES_X[i] = get_arg_val<uint32_t>(start_arg_index + i);
  }
  for (uint32_t i = 0; i < core_grid_y; ++i) {
    PHYSICAL_CORES_Y[i] =
        get_arg_val<uint32_t>(start_arg_index + core_grid_x + i);
  }
}

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t input_dram_addr = get_arg_val<uint32_t>(1);
  uint32_t receiver_sema_addr = get_arg_val<uint32_t>(2);
  uint32_t sender_sema_addr = get_arg_val<uint32_t>(3);
  uint32_t output_dram_addr = get_arg_val<uint32_t>(4);
  init_physical_cores(5);

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

  // Read a single tile from DRAM |input0_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);

  // We use c_in1 for the tile received from other Tensix cores.
  cb_reserve_back(tt::CB::c_in1, /* number of tiles */ 1);
  uint32_t L1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);

  noc_async_read_tile(core_id, bank_for_input, L1_write_addr_in0);
  noc_async_read_barrier();

  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0);
  LOG(DPRINT << "[READER] dram -> cb0: " << *(ptr) << ENDL());
  ptr = reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in0 + 4);
  LOG(DPRINT << "[READER] dram -> cb0: " << *(ptr) << ENDL());

  // ---- Multi-casting start ----
  // Based on multi-casting,
  //  1. Receive i-th tile of input1 matrix from i-th Tensix core.
  //  2. Send |core_id|-th tile of input1 matrix to all other Tensix cores.

  LOG(DPRINT << "[READER] Multicast start" << ENDL());

  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  for (uint32_t i = 0; i < core_id; ++i) {
    noc_semaphore_set(receiver_sema_addr_ptr, 0);

    uint32_t sender_noc_x = PHYSICAL_CORES_X[i % core_grid_x];
    uint32_t sender_noc_y = PHYSICAL_CORES_Y[i / core_grid_x];
    LOG(DPRINT << "[READER] sender_noc_x=" << sender_noc_x << ", "
               << " sender_noc_y=" << sender_noc_y << ENDL());
    uint64_t sender_sema_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_sema_addr);
    noc_semaphore_inc(sender_sema_noc_addr, 1);

    LOG(DPRINT << "[READER] wait " << core_id << ", " << i << ENDL());

    noc_semaphore_wait(receiver_sema_addr_ptr, 1);

#if TINY_DEBUG  // Print first float from CB1 for debugging.
    volatile tt_l1_ptr float* ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1);
    LOG(DPRINT << "[READER] receive cb1: " << *(ptr_first_float) << ENDL());
    ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1 + 4);
    LOG(DPRINT << "[READER] receive cb1: " << *(ptr_first_float) << ENDL());
#endif

    LOG(DPRINT << "[READER] done " << core_id << ", " << i << ENDL());
  }

  LOG(DPRINT << "[READER] sender sema wait " << sender_sema_addr << ENDL());

  volatile tt_l1_ptr uint32_t* sender_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sema_addr);
  noc_semaphore_wait(sender_sema_addr_ptr, number_of_cores - 1);
  noc_semaphore_set(sender_sema_addr_ptr, 0);

  uint64_t multicast_dst_noc_addr = get_noc_multicast_addr(
      PHYSICAL_CORES_X[core_grid_x - 1], PHYSICAL_CORES_Y[core_grid_y - 1], 1,
      1, L1_write_addr_in1);
  // Based on mcast example provided by Tenstorrent, the number of destinations
  // must not include source, since we are NOT really doing a local copy.
  noc_async_write_multicast(L1_write_addr_in0, multicast_dst_noc_addr,
                            tile_size_in_bytes, number_of_cores - 1);

  LOG(DPRINT << "[READER] send sema release" << ENDL());

  *(receiver_sema_addr_ptr) = 1;  // Unlock semaphores of all receivers.
  uint64_t noc_addr = get_noc_multicast_addr(PHYSICAL_CORES_X[core_grid_x - 1],
                                             PHYSICAL_CORES_Y[core_grid_y - 1],
                                             1, 1, receiver_sema_addr);
  // Based on mcast example provided by Tenstorrent, the number of destinations
  // must not include source, since we are NOT really doing a local copy.
  noc_semaphore_set_multicast(receiver_sema_addr, noc_addr,
                              number_of_cores - 1);

  // Note that we do not need both `noc_async_write_barrier()` and
  // `noc_async_writes_flushed()` here. The barrier helps us to guarantee that
  // the multi-cast write is done from receiver side and let it safely start
  // the compute. Since the receiver waits for the semaphore, if it passes the
  // wait line, it means the multi-cast write was already done.

  for (uint32_t i = core_id + 1; i < number_of_cores; ++i) {
    noc_semaphore_set(receiver_sema_addr_ptr, 0);

    uint32_t sender_noc_x = PHYSICAL_CORES_X[i % core_grid_x];
    uint32_t sender_noc_y = PHYSICAL_CORES_Y[i / core_grid_x];
    LOG(DPRINT << "[READER] sender_noc_x=" << sender_noc_x << ", "
               << " sender_noc_y=" << sender_noc_y << ENDL());
    uint64_t sender_sema_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, sender_sema_addr);
    noc_semaphore_inc(sender_sema_noc_addr, 1);

    LOG(DPRINT << "[READER] wait " << core_id << ", " << i << ENDL());

    noc_semaphore_wait(receiver_sema_addr_ptr, 1);

#if TINY_DEBUG  // Print first float from CB1 for debugging.
    volatile tt_l1_ptr float* ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1);
    LOG(DPRINT << "[READER] receive cb1: " << *(ptr_first_float) << ENDL());
    ptr_first_float =
        reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1 + 4);
    LOG(DPRINT << "[READER] receive cb1: " << *(ptr_first_float) << ENDL());
#endif

    LOG(DPRINT << "[READER] done " << core_id << ", " << i << ENDL());
  }

  // ---- Multi-casting end ----
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);
  cb_push_back(tt::CB::c_in1, /* number of tiles */ 1);
}
