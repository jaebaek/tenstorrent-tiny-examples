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

#include "multicast_advanced.h"

#include <cassert>
#include <iostream>
#include <vector>

#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace {

static const uint32_t kSingleTileSize = tiny::SingleTileSize<bfloat16>();

void SetDataMovementKernel(tt::tt_metal::Program& program, CoreCoord core_grid,
                           uint32_t input_device_dram_address,
                           uint32_t receiver_sema_addr,
                           uint32_t sender_sema_addr,
                           uint32_t output_device_dram_address,
                           std::vector<uint32_t> physical_core_coord_info) {
  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
  auto data_mover_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/multicast_advanced.cpp", all_cores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default,
          .compile_args = {core_grid.x, core_grid.y}});

  std::vector<uint32_t> runtime_args{0, input_device_dram_address,
                                     receiver_sema_addr, sender_sema_addr,
                                     output_device_dram_address};
  runtime_args.insert(runtime_args.end(), physical_core_coord_info.begin(),
                      physical_core_coord_info.end());

  uint32_t number_of_cores = core_grid.x * core_grid.y;
  for (uint32_t i = 0; i < number_of_cores; ++i) {
    CoreCoord core = {i % core_grid.x, i / core_grid.x};
    runtime_args[0] = i;
    tt::tt_metal::SetRuntimeArgs(program, data_mover_id, core, runtime_args);
  }
}

std::vector<uint32_t> GetPhysicalCoreCoord(tt::tt_metal::Device* device,
                                           CoreCoord core_grid) {
  std::vector<uint32_t> physical_core_coord_info;
  for (uint32_t x = 0; x < core_grid.x; ++x) {
    CoreCoord core = {x, 0};
    auto core_physical = device->worker_core_from_logical_core(core);
    physical_core_coord_info.push_back(core_physical.x);
  }
  for (uint32_t y = 0; y < core_grid.y; ++y) {
    CoreCoord core = {0, y};
    auto core_physical = device->worker_core_from_logical_core(core);
    physical_core_coord_info.push_back(core_physical.y);
  }
  return std::move(physical_core_coord_info);
}

template <typename T>
tiny::Result _Run(tt::tt_metal::Device* device,
                  std::shared_ptr<tiny::Buffer<T>> input,
                  std::shared_ptr<tiny::Buffer<T>> output) {
  auto core_grid = device->compute_with_storage_grid_size();
  uint32_t num_cores = core_grid.x * core_grid.y;

  tt::tt_metal::CommandQueue& command_queue = device->command_queue();
  tt::tt_metal::Program program{};

  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});

  auto input_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, input->GetSizeInBytes());
  auto output_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, output->GetSizeInBytes());

  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in0, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in1, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in2, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_out0, program, all_cores);

  auto receiver_sema_addr =
      tt::tt_metal::CreateSemaphore(program, all_cores, 0);
  auto sender_sema_addr = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

  SetDataMovementKernel(program, core_grid, input_on_device_dram->address(),
                        receiver_sema_addr, sender_sema_addr,
                        output_on_device_dram->address(),
                        std::move(GetPhysicalCoreCoord(device, core_grid)));

  tt::tt_metal::EnqueueWriteBuffer(command_queue, input_on_device_dram,
                                   input->GetVector().data(), false);
  tt::tt_metal::EnqueueProgram(command_queue, program, false);
  tt::tt_metal::EnqueueReadBuffer(command_queue, output_on_device_dram,
                                  output->GetVector().data(), true);

  return tiny::Result::kSuccess;
}

} /* namespace */

namespace tiny {

template <>
Result MulticastAdvanced<bfloat16>::Run() {
  return _Run<bfloat16>(device_, input_, output_);
}

template <>
Result MulticastAdvanced<float>::Run() {
  return _Run<float>(device_, input_, output_);
}

} /* namespace tiny */
