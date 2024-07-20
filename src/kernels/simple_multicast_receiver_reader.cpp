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
  uint32_t receiver_sema_addr = get_arg_val<uint32_t>(0);

  // Receiver sender's CB::c_in0 tile to CB::c_in1.
  cb_reserve_back(tt::CB::c_in1, /* number of tiles */ 1);
  volatile tt_l1_ptr uint32_t* receiver_sema_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sema_addr);
  noc_semaphore_wait(receiver_sema_addr_ptr, 1);

#if TINY_DEBUG  // Print first float from CB1 for debugging.
  uint32_t L1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
  volatile tt_l1_ptr float* ptr_first_float =
      reinterpret_cast<volatile tt_l1_ptr float*>(L1_write_addr_in1);
  LOG(DPRINT << "[READER] receive cb1: " << *(ptr_first_float) << ENDL());
#endif

  cb_push_back(tt::CB::c_in1, /* number of tiles */ 1);
}
