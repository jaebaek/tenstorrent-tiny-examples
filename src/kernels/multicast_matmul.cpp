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

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "debug/dprint.h"

#define TINY_DEBUG 1

#if TINY_DEBUG
#define LOG(X) DPRINT_MATH(X)
#else
#define LOG(X)
#endif

namespace NAMESPACE {
void MAIN {
  uint32_t number_of_cores = get_arg_val<uint32_t>(0);

  mm_init(tt::CB::c_in0, tt::CB::c_in2, tt::CB::c_out0);
  acquire_dst(tt::DstMode::Full);

  cb_wait_front(tt::CB::c_in0, /* number of tiles */ 1);
  uint32_t cb_in0_ptr = cb_interface[tt::CB::c_in0].fifo_rd_ptr << 4;

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    LOG(DPRINT << "[COMPUTE] loop: " << i << ENDL());
    cb_wait_front(tt::CB::c_in2, /* number of tiles */ 1);

    // ---- can trigger denial of service ----
    copy_tile(tt::CB::c_in0, 0, /* DST */ i);

    cb_reserve_back(tt::CB::c_out1, /* number of tiles */ 1);
    pack_tile(/* DST */ i, tt::CB::c_out1);
    cb_push_back(tt::CB::c_out1, /* number of tiles */ 1);

    copy_tile(tt::CB::c_in2, 0, /* DST */ i);

    cb_reserve_back(tt::CB::c_out2, /* number of tiles */ 1);
    pack_tile(/* DST */ i, tt::CB::c_out2);
    cb_push_back(tt::CB::c_out2, /* number of tiles */ 1);
    // ----

    uint32_t cb_in2_ptr = cb_interface[tt::CB::c_in2].fifo_rd_ptr << 4;

    matmul_tiles(tt::CB::c_in0, tt::CB::c_in2, 0, 0, /* DST */ i, false);
    cb_pop_front(tt::CB::c_in2, /* number of tiles */ 1);
    LOG(DPRINT << "[COMPUTE] loop tail: " << i << ENDL());

    cb_reserve_back(tt::CB::c_out0, /* number of tiles */ 1);
    pack_tile(/* DST */ i, tt::CB::c_out0);
    cb_push_back(tt::CB::c_out0, /* number of tiles */ 1);
  }
  cb_pop_front(tt::CB::c_in0, /* number of tiles */ 1);

  release_dst(tt::DstMode::Full);
}
}  // namespace NAMESPACE
