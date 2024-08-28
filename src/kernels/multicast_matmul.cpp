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
#include "debug/dprint.h"

#define TINY_DEBUG 0

#if TINY_DEBUG
#ifdef UCK_CHLKC_UNPACK
#define LOG(x) DPRINT_UNPACK(DPRINT << "[UNPACK] " << x)
#endif
#ifdef UCK_CHLKC_MATH
#define LOG(x) DPRINT_MATH(DPRINT << "[MATH] " << x)
#endif
#ifdef UCK_CHLKC_PACK
#define LOG(x) DPRINT_PACK(DPRINT << "[PACK] " << x)
#endif
#else
#define LOG(x)
#endif

static inline SliceRange hw_all() {
  return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
}

constexpr uint32_t number_of_cores = get_compile_time_arg_val(0);

namespace NAMESPACE {
void MAIN {
  uint32_t core_id = get_arg_val<uint32_t>(0);

  mm_init();
  cb_wait_front(tt::CB::c_in0, /* number of tiles */ 1);

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    if (i == core_id) {
      cb_wait_front(tt::CB::c_in2, /* number of tiles */ 1);
      cb_wait_front(tt::CB::c_in1, /* number of tiles */ 1);

      tile_regs_acquire();  // Math kernel waits for DEST registers

      matmul_tiles(tt::CB::c_in0, tt::CB::c_in2, 0, 0, 0, false);

      tile_regs_commit();  // Math kernel releases lock for DEST registers

      cb_pop_front(tt::CB::c_in1, /* number of tiles */ 1);
      cb_pop_front(tt::CB::c_in2, /* number of tiles */ 1);
    } else {
      cb_wait_front(tt::CB::c_in1, /* number of tiles */ 1);

      tile_regs_acquire();  // Math kernel waits for DEST registers

      matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);

      tile_regs_commit();  // Math kernel releases lock for DEST registers

      cb_pop_front(tt::CB::c_in1, /* number of tiles */ 1);
    }

    tile_regs_wait();  // Pack kernel wait until Math kernel is done

    cb_reserve_back(tt::CB::c_out0, /* number of tiles */ 1);
    pack_tile(0, tt::CB::c_out0);

#if TINY_DEBUG
    DPRINT_PACK(DPRINT << "[PACK]" << TSLICE(tt::CB::c_out0, 0, hw_all())
                       << ENDL());
#endif

    cb_push_back(tt::CB::c_out0, /* number of tiles */ 1);

    tile_regs_release();  // Pack kernel releases lock for DEST registers
  }

  cb_pop_front(tt::CB::c_in0, /* number of tiles */ 1);
}
}  // namespace NAMESPACE
