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

#include "simple_multicast.h"

#include <variant>

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

namespace {

static const CoreRange kAllCores = {{0, 0}, {0, 3}};
static const CoreCoord kSenderCore = {0, 0};
static const CoreRange kReceiverCores = {{0, 1}, {0, 3}};

template <typename T>
std::shared_ptr<tt::tt_metal::Buffer> CreateSingleTileOnDeviceDRAM(
    tt::tt_metal::Device* device, uint32_t number_of_tiles) {
  tt::tt_metal::InterleavedBufferConfig device_dram_conf{
      .device = device,
      .size = number_of_tiles * tiny::SingleTileSize<T>(),
      .page_size = tiny::SingleTileSize<T>(),
      .buffer_type = tt::tt_metal::BufferType::DRAM};
  return std::move(CreateBuffer(device_dram_conf));
}

template <typename T>
void CreateCircularBufferOnDevice(
    uint32_t circular_buffer_id, tt::tt_metal::Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_range) {
  tt::DataFormat format = tiny::GetDataFormat<T>();
  assert(format != tt::DataFormat::Invalid);

  tt::tt_metal::CircularBufferConfig conf(tiny::SingleTileSize<T>(),
                                          {{circular_buffer_id, format}});
  conf = conf.set_page_size(circular_buffer_id, tiny::SingleTileSize<T>());
  tt::tt_metal::CreateCircularBuffer(program, core_range, conf);
}

void _SetReaderKernel(tt::tt_metal::Program& program,
                      uint32_t receiver_sema_addr,
                      uint32_t input_device_dram_address) {
  auto sender_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/simple_multicast_sender_reader.cpp",
      kSenderCore,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default});

  tt::tt_metal::SetRuntimeArgs(program, sender_id, kSenderCore,
                               {input_device_dram_address, receiver_sema_addr});

  auto receiver_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/simple_multicast_receiver_reader.cpp",
      kReceiverCores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
          .noc = NOC::RISCV_1_default});

  tt::tt_metal::SetRuntimeArgs(program, receiver_id, kReceiverCores,
                               {receiver_sema_addr});
}

void _SetWriteKernel(tt::tt_metal::Program& program,
                     uint32_t output_device_dram_address) {
  auto writer_id = tt::tt_metal::CreateKernel(
      program, "../../src/kernels/simple_multicast_writer.cpp", kAllCores,
      tt::tt_metal::DataMovementConfig{
          .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default});

  for (uint32_t i = 0; i < 4; ++i) {
    CoreCoord core = {0, i};
    tt::tt_metal::SetRuntimeArgs(program, writer_id, core,
                                 {i, output_device_dram_address});
  }
}

void _SetComputeKernel(tt::tt_metal::Program& program) {
  tt::tt_metal::CreateKernel(
      program, "../../src/kernels/simple_multicast_sender.cpp", kSenderCore,
      tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
  tt::tt_metal::CreateKernel(
      program, "../../src/kernels/simple_multicast_receiver.cpp",
      kReceiverCores,
      tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
}

void _SetKernels(tt::tt_metal::Program& program,
                 uint32_t input_device_dram_address,
                 uint32_t receiver_sema_addr,
                 uint32_t output_device_dram_address) {
  _SetReaderKernel(program, input_device_dram_address, receiver_sema_addr);
  _SetWriteKernel(program, output_device_dram_address);
  _SetComputeKernel(program);
}

template <typename T>
tiny::Result _Run(std::shared_ptr<tiny::Buffer<T>> input,
                  std::shared_ptr<tiny::Buffer<T>> output) {
  tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(0);

  tt::tt_metal::CommandQueue& command_queue = device->command_queue();
  tt::tt_metal::Program program{};

  auto input_on_device_dram = CreateSingleTileOnDeviceDRAM<T>(device, 1);
  auto output_on_device_dram = CreateSingleTileOnDeviceDRAM<T>(device, 4);

  CreateCircularBufferOnDevice<T>(tt::CB::c_in0, program, kSenderCore);
  CreateCircularBufferOnDevice<T>(tt::CB::c_in1, program, kAllCores);
  CreateCircularBufferOnDevice<T>(tt::CB::c_out0, program, kAllCores);

  auto receiver_sema_addr =
      tt::tt_metal::CreateSemaphore(program, kAllCores, 0);

  _SetKernels(program, input_on_device_dram->address(), receiver_sema_addr,
              output_on_device_dram->address());

  tt::tt_metal::EnqueueWriteBuffer(command_queue, input_on_device_dram,
                                   input->GetVector().data(), false);
  tt::tt_metal::EnqueueProgram(command_queue, program, false);
  tt::tt_metal::EnqueueReadBuffer(command_queue, output_on_device_dram,
                                  output->GetVector().data(), true);

  bool pass = tt::tt_metal::CloseDevice(device);
  return pass ? tiny::Result::kSuccess : tiny::Result::kFail;
}

} /* namespace */

namespace tiny {

template <>
Result SimpleMulticast<bfloat16>::Run() {
  return _Run<bfloat16>(input_, output_);
}

template <>
Result SimpleMulticast<float>::Run() {
  return _Run<float>(input_, output_);
}

} /* namespace tiny */