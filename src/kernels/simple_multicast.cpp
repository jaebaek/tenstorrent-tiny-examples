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

#include "compute_kernel_api.h"
#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif

#define TINY_DEBUG 1

#if TINY_DEBUG
#define LOG(X) DPRINT_MATH(X)
#else
#define LOG(X)
#endif

static inline SliceRange hw_all() {
  return SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
}

namespace NAMESPACE {
void MAIN {
  acquire_dst(tt::DstMode::Tile);

  cb_wait_front(tt::CB::c_in0, /* number of tiles */ 1);
  copy_tile_to_dst_init_short();
  copy_tile(tt::CB::c_in0, 0, /* DST */ 0);
#if TINY_DEBUG
  DPRINT_UNPACK(DPRINT << TSLICE(tt::CB::c_in0, 0, hw_all()) << ENDL());
#endif
  cb_pop_front(tt::CB::c_in0, /* number of tiles */ 1);

  MATH((llk_math_matmul_init<MATH_FIDELITY>(tt::CB::c_in0, tt::CB::c_in1, 0)));

  // PACK(( llk_pack_hw_configure_disaggregated<false,
  // DST_ACCUM_MODE>(tt::CB::c_out0) ));
  PACK((llk_pack_init(tt::CB::c_out0)));
  // PACK(( llk_setup_outputs()  ));
  PACK((llk_pack_dest_init<false, DST_ACCUM_MODE>(tt::CB::c_out0)));

  LOG(DPRINT << "[COMPUTE] pack tile" << ENDL());

  cb_reserve_back(tt::CB::c_out0, /* number of tiles */ 1);
  pack_tile(/* DST */ 0, tt::CB::c_out0);

#if TINY_DEBUG
  DPRINT_PACK(DPRINT << TSLICE(tt::CB::c_out0, 0, hw_all()) << ENDL());
#endif
  cb_push_back(tt::CB::c_out0, /* number of tiles */ 1);

  release_dst(tt::DstMode::Tile);
  LOG(DPRINT << "[COMPUTE] done" << ENDL());
}
}  // namespace NAMESPACE
