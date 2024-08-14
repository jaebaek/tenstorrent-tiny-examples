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

static inline SliceRange hw_all() {
  return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
}

constexpr uint32_t core_grid_x = get_compile_time_arg_val(0);
constexpr uint32_t core_grid_y = get_compile_time_arg_val(1);
constexpr uint32_t number_of_cores = core_grid_x * core_grid_y;
uint32_t PHYSICAL_CORES_X[core_grid_x];
uint32_t PHYSICAL_CORES_Y[core_grid_y];
uint32_t physical_core_x_start, physical_core_y_start;
uint32_t physical_core_x_end, physical_core_y_end;

constexpr uint32_t tile_size_in_bytes = get_tile_size(tt::CB::c_in0);
constexpr DataFormat format = get_dataformat(tt::CB::c_in0);

static void init_physical_cores(const uint32_t start_arg_index) {
  for (uint32_t i = 0; i < core_grid_x; ++i) {
    PHYSICAL_CORES_X[i] = get_arg_val<uint32_t>(start_arg_index + i);
  }
  for (uint32_t i = 0; i < core_grid_y; ++i) {
    PHYSICAL_CORES_Y[i] =
        get_arg_val<uint32_t>(start_arg_index + core_grid_x + i);
  }
  physical_core_x_start = PHYSICAL_CORES_X[0];
  physical_core_y_start = PHYSICAL_CORES_Y[0];
  physical_core_x_end = PHYSICAL_CORES_X[core_grid_x - 1];
  physical_core_y_end = PHYSICAL_CORES_Y[core_grid_y - 1];
}

static inline void send(uint32_t core_id, uint32_t src, uint32_t dest,
                        uint32_t receiver_sema_addr,
                        uint32_t sender_sema_addr) {
#if TINY_DEBUG
  LOG(DPRINT << TSLICE(tt::CB::c_in1, 0, hw_all()) << ENDL());
#endif

  volatile tt_l1_ptr uint32_t* sender_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sema_addr);
  noc_semaphore_wait(sender_sema_addr_ptr, number_of_cores - 1);
  noc_semaphore_set(sender_sema_addr_ptr, 0);

  uint64_t multicast_dst_noc_addr = get_noc_multicast_addr(
      physical_core_x_end, physical_core_y_end, physical_core_x_start,
      physical_core_y_start, dest);
  noc_async_write_multicast(src, multicast_dst_noc_addr, tile_size_in_bytes,
                            number_of_cores - 1);

  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  *(receiver_sema_addr_ptr) = 1;  // Unlock semaphores of all receivers.
  uint64_t noc_addr = get_noc_multicast_addr(
      physical_core_x_end, physical_core_y_end, physical_core_x_start,
      physical_core_y_start, receiver_sema_addr);
  noc_semaphore_set_multicast(receiver_sema_addr, noc_addr,
                              number_of_cores - 1);

  uint32_t L1_read_addr_in0 = get_read_ptr(tt::CB::c_in1);
#if TINY_DEBUG
  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_read_addr_in0);
  LOG(DPRINT << *ptr << ENDL());
  ptr = reinterpret_cast<volatile tt_l1_ptr float*>(L1_read_addr_in0 + 4);
  LOG(DPRINT << *ptr << ENDL());
  LOG(DPRINT << "[READER] write to " << (core_id * number_of_cores + core_id)
             << ENDL());
#endif

  // We have to re-initialize receiver sema for its next receiver turn.
  *(receiver_sema_addr_ptr) = 0;

  LOG(DPRINT << "[READER] done" << ENDL());
}

static inline void receive(uint32_t core_id, uint32_t receiver_sema_addr,
                           uint32_t sender_sema_addr, uint32_t sender) {
  uint32_t sender_noc_x = PHYSICAL_CORES_X[sender % core_grid_x];
  uint32_t sender_noc_y = PHYSICAL_CORES_Y[sender / core_grid_x];
  LOG(DPRINT << "[READER] sender_noc_x=" << sender_noc_x << ", "
             << " sender_noc_y=" << sender_noc_y << ENDL());
  uint64_t sender_sema_noc_addr =
      get_noc_addr(sender_noc_x, sender_noc_y, sender_sema_addr);
  noc_semaphore_inc(sender_sema_noc_addr, 1);

  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  noc_semaphore_wait(receiver_sema_addr_ptr, 1);
  noc_semaphore_set(receiver_sema_addr_ptr, 0);

#if TINY_DEBUG
  LOG(DPRINT << TSLICE(tt::CB::c_in2, 0, hw_all()) << ENDL());
#endif

  uint32_t L1_read_addr_in1 = get_read_ptr(tt::CB::c_in2);
#if TINY_DEBUG
  volatile tt_l1_ptr float* ptr =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_read_addr_in1);
  LOG(DPRINT << *ptr << ENDL());
  ptr = reinterpret_cast<volatile tt_l1_ptr float*>(L1_read_addr_in1 + 4);
  LOG(DPRINT << *ptr << ENDL());
  LOG(DPRINT << "[READER] write to " << (core_id * number_of_cores + sender)
             << ENDL());
#endif

  LOG(DPRINT << "[READER] done" << ENDL());
}

void kernel_main() {
  uint32_t core_id = get_arg_val<uint32_t>(0);
  uint32_t input0_dram_addr = get_arg_val<uint32_t>(1);
  uint32_t input1_dram_addr = get_arg_val<uint32_t>(2);
  uint32_t receiver_sema_addr = get_arg_val<uint32_t>(3);
  uint32_t sender_sema_addr = get_arg_val<uint32_t>(4);
  init_physical_cores(5);

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input0 = {
      .bank_base_address = input0_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

  // Read a single tile from DRAM |input0_dram_addr| to circular buffer in0.
  cb_reserve_back(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t L1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
  noc_async_read_tile(core_id, bank_for_input0, L1_write_addr_in0);
  noc_async_read_barrier();
  cb_push_back(tt::CB::c_in0, /* number of tiles */ 1);

  const InterleavedAddrGenFast</* From DRAM address */ true> bank_for_input1 = {
      .bank_base_address = input1_dram_addr,
      .page_size = tile_size_in_bytes,
      .data_format = format};

  // Read a single tile from DRAM |input1_dram_addr| to circular buffer in1.
  cb_reserve_back(tt::CB::c_in1, /* number of tiles */ 1);
  uint32_t L1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
  noc_async_read_tile(core_id, bank_for_input1, L1_write_addr_in1);
  noc_async_read_barrier();

  cb_reserve_back(tt::CB::c_in2, /* number of tiles */ 1);
  uint32_t L1_write_addr_in2 = get_write_ptr(tt::CB::c_in2);

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    if (i == core_id) {
      send(core_id, L1_write_addr_in1, L1_write_addr_in2, receiver_sema_addr,
           sender_sema_addr);
    } else {
      receive(core_id, receiver_sema_addr, sender_sema_addr, i);
    }
  }

  cb_push_back(tt::CB::c_in2, /* number of tiles */ 1);
  cb_push_back(tt::CB::c_in1, /* number of tiles */ 1);
}
