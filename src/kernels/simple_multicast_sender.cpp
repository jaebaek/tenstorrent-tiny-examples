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
  copy_tile_init();
  acquire_dst(tt::DstMode::Tile);

  cb_wait_front(tt::CB::c_in0, /* number of tiles */ 1);
#if TINY_DEBUG
  DPRINT_UNPACK(DPRINT << TSLICE(tt::CB::c_in0, 0, hw_all()) << ENDL());
#endif
  copy_tile(tt::CB::c_in0, 0, /* DST */ 0);
  cb_pop_front(tt::CB::c_in0, /* number of tiles */ 1);

  LOG(DPRINT << "[COMPUTE] pack tile" << ENDL());

  mm_init();

  cb_reserve_back(tt::CB::c_out0, /* number of tiles */ 1);
  pack_tile(/* DST */ 0, tt::CB::c_out0);
#if TINY_DEBUG
  DPRINT_PACK(DPRINT << TSLICE(tt::CB::c_out0, 0, hw_all()) << ENDL());
#endif
  cb_push_back(tt::CB::c_out0, /* number of tiles */ 1);

  LOG(DPRINT << "[COMPUTE] done" << ENDL());

  release_dst(tt::DstMode::Tile);
}
}  // namespace NAMESPACE
