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

#include "multicast_matmul.h"

#include <cassert>
#include <iostream>
#include <vector>

#include "tt_metal/host_api.hpp"
#include "utils.h"

namespace {

static const uint32_t kSingleTileSize = tiny::SingleTileSize<bfloat16>();

void SetReaderKernel(tt::tt_metal::Program& program, CoreCoord core_grid,
                     uint32_t input0_device_dram_address,
                     uint32_t input1_device_dram_address,
                     uint32_t receiver_sema_addr, uint32_t sender_sema_addr,
                     std::vector<uint32_t> physical_core_coord_info) {
  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
  auto reader_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/multicast_matmul_reader.cpp", all_cores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default,
          .compile_args = {core_grid.x, core_grid.y}});

  std::vector<uint32_t> runtime_args{0, input0_device_dram_address,
                                     input1_device_dram_address,
                                     receiver_sema_addr, sender_sema_addr};
  runtime_args.insert(runtime_args.end(), physical_core_coord_info.begin(),
                      physical_core_coord_info.end());

  uint32_t number_of_cores = core_grid.x * core_grid.y;
  for (uint32_t i = 0; i < number_of_cores; ++i) {
    CoreCoord core = {i % core_grid.x, i / core_grid.x};
    runtime_args[0] = i;
    tt::tt_metal::SetRuntimeArgs(program, reader_id, core, runtime_args);
  }
}

void SetWriteKernel(tt::tt_metal::Program& program, CoreCoord core_grid,
                    uint32_t output_device_dram_address) {
  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
  uint32_t number_of_cores = core_grid.x * core_grid.y;

  auto writer_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/multicast_matmul_writer.cpp", all_cores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default,
          .compile_args = {number_of_cores}});

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    CoreCoord core = {i % core_grid.x, i / core_grid.x};
    tt::tt_metal::SetRuntimeArgs(program, writer_id, core,
                                 {i, output_device_dram_address});
  }
}

void SetComputeKernel(tt::tt_metal::Program& program, CoreCoord core_grid) {
  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
  uint32_t number_of_cores = core_grid.x * core_grid.y;

  auto compute_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/multicast_matmul.cpp", all_cores,
      tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                  .compile_args = {number_of_cores}});

  for (uint32_t i = 0; i < number_of_cores; ++i) {
    CoreCoord core = {i % core_grid.x, i / core_grid.x};
    tt::tt_metal::SetRuntimeArgs(program, compute_id, core, {i});
  }
}

void SetKernels(tt::tt_metal::Program& program, CoreCoord core_grid,
                uint32_t input0_device_dram_address,
                uint32_t input1_device_dram_address,
                uint32_t receiver_sema_addr, uint32_t sender_sema_addr,
                uint32_t output_device_dram_address,
                std::vector<uint32_t> physical_core_coord_info) {
  SetReaderKernel(program, core_grid, input0_device_dram_address,
                  input1_device_dram_address, receiver_sema_addr,
                  sender_sema_addr, std::move(physical_core_coord_info));
  // SetWriteKernel(program, core_grid, output_device_dram_address);
  // SetComputeKernel(program, core_grid);
}

template <typename T>
tiny::Result _Run(tt::tt_metal::Device* device,
                  std::shared_ptr<tiny::Buffer<T>> input0,
                  std::shared_ptr<tiny::Buffer<T>> input1,
                  std::shared_ptr<tiny::Buffer<T>> output) {
  CoreCoord core_grid = device->compute_with_storage_grid_size();
  uint32_t num_cores = core_grid.x * core_grid.y;

  input0->Tilize(tiny::TileWidth(), num_cores * tiny::TileHeight());
  input1->Tilize(num_cores * tiny::TileHeight(), tiny::TileWidth());

  tt::tt_metal::CommandQueue& command_queue = device->command_queue();
  tt::tt_metal::Program program{};

  auto input0_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, input0->GetSizeInBytes());
  auto input1_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, input1->GetSizeInBytes());
  auto output_on_device_dram =
      tiny::CreateBufferOnDeviceDRAM<T>(device, output->GetSizeInBytes());

  auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in0, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in1, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_in2, program, all_cores);
  tiny::CreateCircularBufferOnDevice<T>(tt::CB::c_out0, program, all_cores);

  auto receiver_sema_addr =
      tt::tt_metal::CreateSemaphore(program, all_cores, 0);
  auto sender_sema_addr = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

  SetKernels(program, core_grid, input0_on_device_dram->address(),
             input1_on_device_dram->address(), receiver_sema_addr,
             sender_sema_addr, output_on_device_dram->address(),
             std::move(tiny::GetPhysicalCoreCoord(device, core_grid)));

  tt::tt_metal::EnqueueWriteBuffer(command_queue, input0_on_device_dram,
                                   input0->GetVector().data(), false);
  tt::tt_metal::EnqueueWriteBuffer(command_queue, input1_on_device_dram,
                                   input1->GetVector().data(), false);
  tt::tt_metal::EnqueueProgram(command_queue, program, false);
  tt::tt_metal::Finish(command_queue);

  tt::tt_metal::EnqueueReadBuffer(command_queue, output_on_device_dram,
                                  output->GetVector().data(), true);

  return tiny::Result::kSuccess;
}

} /* namespace */

namespace tiny {

template <>
Result MulticastMatrixMultiplication<bfloat16>::Run() {
  return _Run<bfloat16>(device_, inputs_[0], inputs_[1], output_);
}

template <>
Result MulticastMatrixMultiplication<float>::Run() {
  return _Run<float>(device_, inputs_[0], inputs_[1], output_);
}

} /* namespace tiny */
